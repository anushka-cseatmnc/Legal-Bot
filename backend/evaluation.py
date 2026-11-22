"""Reproduce retrieval measurements and record actual local generation responses."""

import argparse
import json
import math
import os
import statistics
import threading
import time
from dataclasses import replace
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma

from backend.db import (
    BGE_MODEL, COLLECTION_NAME, EMBEDDING_MODEL, PROJECT_ROOT, get_embeddings,
    get_vector_store, index_root, project_path, validate_collection,
)
from backend.rag_pipeline import LegalAssistant, GenerationTimeout, ProviderUnavailable
from backend.retrieval import LegalRetriever, get_reranker
from config import Settings

RTI = "RTI act, 2005 (amended)-english version.pdf"
CRPC = "the_code_of_criminal_procedure,_1973.pdf"


def case(id_, question, file=None, section=None, phrase=None, *,
         history=None, status=None, pages=None, **options):
    return dict(id=id_, question=question, file=file, section=section, phrase=phrase,
                history=history or [], expected_status=status, pages=pages, options=options)


AUDIT = [
    case("01", "When must a public information officer answer an RTI request affecting life or liberty?",
         RTI, "7", "forty-eight hours", pages=[7]),
    case("02", "What is the usual time limit for a consumer complaint, and can delay be excused?",
         "consumer protection.pdf", "69", "sufficient cause", pages=[28]),
    case("03", "What minimum ages does the Hindu Marriage Act prescribe?",
         "The Hindu marriage act.pdf", "5", "bridegroom has completed", pages=[5]),
    case("04", "What does Article 21 protect?",
         "constitution of india.pdf", "21", "life or personal liberty", pages=[42]),
    case("05", "Which IT Act section covers identity theft?",
         "it_act_2000_updated.pdf", "66C", "electronic signature", pages=[24]),
    case("06", "In August 2025, what can I do if police refuse to record a cognizable offence?",
         "BNSS 2023.pdf", "173", "superintendent of police", pages=[53]),
    case("07", "Under the old CrPC what section governed FIRs?",
         CRPC, "154", "officer in charge", pages=[78]),
    case("08", "Which section governs admissibility of electronic records under the BSA?",
         "BSA 2023.pdf", "63", "information contained in an electronic record", pages=[22, 23]),
    case("09", "What is the statute of limitations for divorce in California?", status="out_of_scope"),
    case("10", "What is the weather in Pune tomorrow?", status="out_of_scope"),
    case("11", "What if my life is at risk?", RTI, "7", "forty-eight hours",
         history=["When is an RTI reply due?"], pages=[7]),
]
HELD_OUT = [
    case("H1", "Can police receive information about a cognizable offence electronically under BNSS section 173?",
         "BNSS 2023.pdf", "173", "electronic communication", pages=[52]),
    case("H2", "Do I need a certificate for an electronic record under BSA section 63?",
         "BSA 2023.pdf", "63", "certificate", pages=[22, 23]),
    case("H3", "What does BNS section 358 say about the repeal of the IPC?",
         "BNS 2023.pdf", "358", "indian penal code", pages=[102]),
    case("H4", "My criminal investigation started in 2023. Which procedure applies now?", status="insufficient"),
    case("H5", "What is the latest cricket score?", status="out_of_scope"),
    case("H6", "What law applies to evidence in an Indian trial already pending when BSA commenced?",
         "BSA 2023.pdf", "170", "indian evidence act", pages=[45]),
]
DEPLOYMENT_CASES = [
    case("D1", "What guarantee does Article 14 of the Constitution provide?",
         "constitution of india.pdf", "14", "equal protection of the laws", pages=[37]),
    case("D2", "Can a recognised consumer association file a complaint under section 35 of the Consumer Protection Act?",
         "consumer protection.pdf", "35", "recognised consumer association", pages=[18]),
    case("D3", "What separation period does Hindu Marriage Act section 13B require for divorce by mutual consent?",
         "The Hindu marriage act.pdf", "13B", "one year or more", pages=[8]),
    case("D4", "What does IT Act section 66D say about cheating by personation through a computer?",
         "it_act_2000_updated.pdf", "66D", "computer resource cheats by personation", pages=[24]),
    case("D5", "What does section 303 of the BNS require for theft of movable property?",
         "BNS 2023.pdf", "303", "dishonestly any movable property", pages=[78]),
    case("D6", "Who hears my first appeal under RTI Act section 19 if I did not receive an answer?",
         RTI, "19", "senior in rank", pages=[15]),
]


class MemorySampler:
    def __init__(self, ollama=False):
        self.ollama = ollama
        self.stop = threading.Event()
        self.peak_rss = 0
        self.peak_ollama = 0
        self.worker = threading.Thread(target=self.sample, daemon=True)

    def sample(self):
        import psutil

        self_process = psutil.Process()
        while not self.stop.is_set():
            self.peak_rss = max(self.peak_rss, self_process.memory_info().rss)
            if self.ollama:
                total = 0
                for proc in psutil.process_iter(["name", "memory_info"], ad_value=None):
                    info = proc.info
                    if (info["name"] or "").casefold() in {
                        "ollama", "ollama.exe", "ollama app.exe", "llama-server", "llama-server.exe",
                    } and info["memory_info"]:
                        total += info["memory_info"].rss
                self.peak_ollama = max(self.peak_ollama, total)
            self.stop.wait(.4)

    def __enter__(self):
        self.worker.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.worker.join()

    def report(self):
        return {"process_peak_rss_mib": round(self.peak_rss / 2**20, 1),
                "ollama_peak_combined_rss_mib": round(self.peak_ollama / 2**20, 1) if self.ollama else None}


def retrieval_row(fixture, status, documents, elapsed):
    matching = [(i, doc) for i, doc in enumerate(documents, 1)
                if doc.metadata.get("document") == fixture["file"]
                and doc.metadata.get("section") == fixture["section"]]
    source_pages = fixture["pages"] or []
    passage = [(i, doc) for i, doc in matching if fixture["phrase"] in
               " ".join(doc.page_content.casefold().split())]
    return {
        "id": fixture["id"], "status": status, "ms": round(elapsed * 1000, 2),
        "provision_rank": matching[0][0] if matching else None,
        "passage_rank": passage[0][0] if passage else None,
        "page_matches": [doc.metadata.get("pdf_page") in source_pages for _, doc in passage],
        "ranked_sources": [{key: doc.metadata.get(key) for key in ("document", "section", "pdf_page", "legal_status")}
                           for doc in documents],
    }


def metrics(fixtures, rows):
    targets = [row for fixture, row in zip(fixtures, rows) if fixture["file"]]
    statuses = [(fixture, row) for fixture, row in zip(fixtures, rows) if fixture["expected_status"]]
    timings = sorted(row["ms"] for row in rows)
    return {
        "target_cases": len(targets),
        **{f"{rank}_at_{k}": sum(row[rank] is not None and row[rank] <= k for row in targets)
           for rank in ("provision_rank", "passage_rank") for k in (2, 5)},
        "expected_status_matches": sum(row["status"] == fixture["expected_status"] for fixture, row in statuses),
        "expected_status_cases": len(statuses),
        "passage_metadata_correct": sum(correct for row in targets for correct in row["page_matches"]),
        "passage_metadata_checks": sum(len(row["page_matches"]) for row in targets),
        "median_ms": statistics.median(timings),
        "p95_ms": timings[math.ceil(.95 * len(timings)) - 1],
    }


def run(args):
    hosted = args.profile == "cloud"
    if not hosted:
        import torch

        settings = Settings.from_env()
        torch.set_num_threads(settings.threads)
    root = project_path(args.index) if args.index else index_root()
    model = BGE_MODEL if args.profile == "bge" else EMBEDDING_MODEL
    if args.profile == "bge" and not args.index:
        root /= "bge"
    started = time.perf_counter()
    if hosted:
        from backend.cloud_service import CloudSettings, create_retriever

        settings = CloudSettings.from_mapping({})
        if args.hosted_answer and not settings.provider.api_key:
            raise SystemExit("Set GEMINI_API_KEY locally to run live hosted inference.")
        retriever = create_retriever(settings)
    elif args.profile == "baseline":
        path = root / "snapshots" / args.baseline_snapshot
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        client = validate_collection(path, manifest["chunks"])
        store = Chroma(client=client, collection_name=COLLECTION_NAME, embedding_function=get_embeddings(),
                       create_collection_if_not_exists=False)
        legacy = store.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 10})
        ranked_two = store.as_retriever(search_type="mmr", search_kwargs={"k": 2, "fetch_k": 4})
    else:
        store = get_vector_store(root, model)
        retriever = LegalRetriever(store, candidates=settings.candidates, top_k=5,
                                   reranker=get_reranker() if args.profile in {"rerank", "bge"} else None)
    initialization_sec = round(time.perf_counter() - started, 3)
    cases = AUDIT + HELD_OUT + (DEPLOYMENT_CASES if hosted else [])
    if args.case:
        cases = [fixture for fixture in cases if fixture["id"] in args.case]
    rows = []
    if args.generate or args.hosted_answer:
        if hosted:
            from backend.hosted import Gemini
            assistant = LegalAssistant(Settings(), retriever=retriever, provider=Gemini(settings.provider))
        else:
            settings = replace(settings, model=args.generate, fallback_model="", cloud_model="")
            assistant = LegalAssistant(settings, retriever=retriever)
        try:
            for fixture in cases:
                t = time.perf_counter()
                try:
                    output = assistant.answer(fixture["question"], history=fixture["history"], **fixture["options"])
                except (ProviderUnavailable, GenerationTimeout) as exc:
                    output = {"status": "error", "error": type(exc).__name__}
                rows.append({"id": fixture["id"], "ms": round((time.perf_counter() - t) * 1000, 2), **output})
                with args.output.with_suffix(".jsonl").open("a", encoding="utf-8") as record:
                    record.write(json.dumps(rows[-1], ensure_ascii=False) + "\n")
                print(fixture["id"], output["status"], rows[-1]["ms"], flush=True)
                if args.hosted_answer and args.pace_sec and fixture != cases[-1]:
                    time.sleep(args.pace_sec)
        finally:
            if hosted:
                assistant.provider.client.close()
            else:
                assistant.provider.close()
        return {"initialization_sec": initialization_sec, "cases": rows,
                "hosted_model": settings.provider.model if hosted else None}

    for fixture in cases:
        t = time.perf_counter()
        if args.profile == "baseline":
            docs = legacy.invoke(fixture["question"])
            elapsed = time.perf_counter() - t
            top2 = retrieval_row(fixture, "unfiltered", ranked_two.invoke(fixture["question"]), elapsed)
            row = retrieval_row(fixture, "unfiltered", docs, elapsed)
            row["baseline_at2"] = top2
        else:
            result = retriever.retrieve(fixture["question"], fixture["history"], **fixture["options"])
            row = retrieval_row(fixture, result.status, result.documents, time.perf_counter() - t)
        rows.append(row)
    end = len(AUDIT) + len(HELD_OUT)
    groups = {"audit": (AUDIT, rows[:len(AUDIT)]), "held_out": (HELD_OUT, rows[len(AUDIT):end])}
    if hosted:
        groups["deployment"] = (DEPLOYMENT_CASES, rows[end:])
    summaries = {group: metrics(fixtures, selected) for group, (fixtures, selected) in groups.items()}
    if args.profile == "baseline":
        for group, (fixtures, selected) in groups.items():
            measured = metrics(fixtures, [row["baseline_at2"] for row in selected])
            for rank in ("provision_rank_at_2", "passage_rank_at_2"):
                summaries[group][rank] = measured[rank]
    return {"initialization_sec": initialization_sec, **summaries, "cases": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["baseline", "hybrid", "rerank", "bge", "cloud"], required=True)
    parser.add_argument("--index")
    parser.add_argument("--baseline-snapshot")
    parser.add_argument("--generate", help="Record actual generation with this provisioned Ollama tag")
    parser.add_argument("--hosted-answer", action="store_true", help="Send benchmark questions to the configured Gemini project")
    parser.add_argument("--pace-sec", type=int, default=16, help="Wait between hosted benchmark cases")
    parser.add_argument("--case", action="append", help="Run just this generation case. Repeat to select more cases.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.profile != "cloud":
        load_dotenv(PROJECT_ROOT / ".env")
    if args.profile == "baseline" and (not args.baseline_snapshot or args.generate):
        parser.error("Baseline requires a saved snapshot and supports retrieval only.")
    if (args.hosted_answer and args.profile != "cloud") or (args.generate and args.profile == "cloud"):
        parser.error("Use --hosted-answer only with the cloud profile.")
    cases = AUDIT + HELD_OUT + (DEPLOYMENT_CASES if args.profile == "cloud" else [])
    if args.case and (not (args.generate or args.hosted_answer) or set(args.case) - {fixture["id"] for fixture in cases}):
        parser.error("--case requires generation and a known case ID.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.generate or args.hosted_answer:
        args.output.with_suffix(".jsonl").write_text("", encoding="utf-8")
    with MemorySampler(bool(args.generate)) as memory:
        result = run(args)
    result.update(memory.report(), profile=args.profile, model=args.generate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "cases"}, indent=2))


if __name__ == "__main__":
    main()
