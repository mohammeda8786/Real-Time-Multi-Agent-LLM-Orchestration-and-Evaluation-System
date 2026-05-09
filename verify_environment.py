import importlib
import inspect
import os
import sys
import tempfile
import time

CHECKS = []


def check(name):
    def decorator(fn):
        CHECKS.append((name, fn))
        return fn
    return decorator


@check("Python version")
def check_python_version():
    v = sys.version_info
    ver_str = f"Python {sys.version.split()[0]}"
    if v < (3, 11):
        return False, f"{ver_str} — [ERROR] Need 3.11+ (recommend 3.11 or 3.12)."
    if v >= (3, 13):
        return (
            True,
            f"{ver_str} — [WARN] Newer than tested 3.11–3.12; Groq/Chroma may warn. Prefer 3.11 or 3.12.",
        )
    return True, f"{ver_str} — [OK] Supported (3.11–3.12)."


@check("Groq SDK")
def check_groq():
    try:
        groq = importlib.import_module("groq")
        version = getattr(groq, "__version__", "unknown")
        AsyncGroq = getattr(groq, "AsyncGroq", None)
        if AsyncGroq is None:
            return False, f"groq {version} imported but AsyncGroq not found"

        signature = inspect.signature(AsyncGroq.__init__)
        supports_proxies = "proxies" in signature.parameters
        details = [f"groq {version}", f"AsyncGroq accepts proxies={supports_proxies}"]
        return True, "; ".join(details)
    except Exception as exc:
        return False, f"Groq import failed: {type(exc).__name__}: {exc}"


@check("HTTPX compatibility")
def check_httpx():
    try:
        httpx = importlib.import_module("httpx")
        version = getattr(httpx, "__version__", "unknown")
        signature = inspect.signature(httpx.AsyncClient)
        supports_proxies = "proxies" in signature.parameters
        return True, f"httpx {version}; AsyncClient proxies={supports_proxies}"
    except Exception as exc:
        return False, f"httpx import failed: {type(exc).__name__}: {exc}"


@check("ChromaDB")
def check_chromadb():
    try:
        chromadb = importlib.import_module("chromadb")
        version = getattr(chromadb, "__version__", "unknown")
        path = os.path.join(tempfile.gettempdir(), "mega_ai_chroma_check")
        client = None
        try:
            client = chromadb.PersistentClient(path=path)
            close = getattr(client, "close", None)
            if callable(close):
                close()
        except AttributeError:
            client = chromadb.Client()
            close = getattr(client, "close", None)
            if callable(close):
                close()
        return True, f"chromadb {version} initialized successfully"
    except Exception as exc:
        return False, f"ChromaDB import/init failed: {type(exc).__name__}: {exc}"


@check("Pydantic")
def check_pydantic():
    try:
        pydantic = importlib.import_module("pydantic")
        version = getattr(pydantic, "__version__", "unknown")
        if version.startswith("1."):
            return False, f"Unsupported pydantic version: {version}. Use pydantic>=2.7,<3"
        return True, f"pydantic {version}"
    except Exception as exc:
        return False, f"Pydantic import failed: {type(exc).__name__}: {exc}"


@check("Sentence Transformers")
def check_sentence_transformers():
    try:
        from sentence_transformers import SentenceTransformer

        model_name = "all-MiniLM-L6-v2"
        start_time = time.time()
        model = SentenceTransformer(model_name)
        dimension = model.get_sentence_embedding_dimension()
        latency = time.time() - start_time
        return True, f"{model_name} loaded dim={dimension} load_time={latency:.2f}s"
    except Exception as exc:
        return False, f"Embedding load failed: {type(exc).__name__}: {exc}"


def main():
    print("Environment validation for Mega.AI")
    print("=" * 50)
    success = True
    for name, fn in CHECKS:
        ok, detail = fn()
        status = "OK" if ok else "FAIL"
        print(f"{status:4} | {name:25} | {detail}")
        if not ok:
            success = False
    print("=" * 50)
    if not success:
        print("One or more startup dependency checks failed.")
        print("Review the output above and install compatible versions.")
        sys.exit(1)
    print("All dependency checks passed.")


if __name__ == "__main__":
    main()
