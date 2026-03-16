import concurrent.futures
import logging

from batch_run import run_batch

logger = logging.getLogger(__name__)


def process_audio_job(job_id, input_length, sr, dsp_only, qa_only, no_parallel):
    logger.info("[SubAgent] Starte Job %s", job_id)
    # Hier könnte ein echtes Audiosignal geladen werden
    # Für Demo: run_batch mit Dummy-Parametern
    run_batch(
        input_length=input_length,
        sr=sr,
        dsp_only=dsp_only,
        qa_only=qa_only,
        no_parallel=no_parallel,
    )
    logger.info("[SubAgent] Job %s fertig", job_id)


def main():
    # Beispiel: 4 parallele Jobs mit unterschiedlichen Parametern
    jobs = [
        {
            "job_id": 1,
            "input_length": 4096,
            "sr": 16000,
            "dsp_only": False,
            "qa_only": False,
            "no_parallel": False,
        },
        {
            "job_id": 2,
            "input_length": 8192,
            "sr": 44100,
            "dsp_only": True,
            "qa_only": False,
            "no_parallel": True,
        },
        {
            "job_id": 3,
            "input_length": 4096,
            "sr": 16000,
            "dsp_only": False,
            "qa_only": True,
            "no_parallel": False,
        },
        {
            "job_id": 4,
            "input_length": 2048,
            "sr": 22050,
            "dsp_only": False,
            "qa_only": False,
            "no_parallel": True,
        },
    ]
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_audio_job, **job) for job in jobs]
        for future in concurrent.futures.as_completed(futures):
            future.result()
    logger.info("[SubAgent] Alle Jobs abgeschlossen.")


if __name__ == "__main__":
    main()
