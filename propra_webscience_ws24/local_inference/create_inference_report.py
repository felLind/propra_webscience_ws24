from sklearn.metrics import accuracy_score, classification_report
from tabulate import tabulate  # type: ignore

from propra_webscience_ws24.constants import RESULTS_PATH
from propra_webscience_ws24.local_inference.deepseek import (
    InferenceResults,
    EnrichedModelOutput,
)

REPORT_WIDTH = 120

DEEPSEEK_MODEL_NAME_1_5B = "deepseek-r1-1.5b"
DEEPSEEK_MODEL_NAME_8B = "deepseek-r1-8b"
DEEPSEEK_MODEL_NAME_32B = "deepseek-r1-32b"
DEEPSEEK_MODEL_NAME_70B = "deepseek-r1-70b"

DEEPSEEK_MODEL_NAMES = [
    DEEPSEEK_MODEL_NAME_1_5B,
    DEEPSEEK_MODEL_NAME_8B,
    DEEPSEEK_MODEL_NAME_32B,
    DEEPSEEK_MODEL_NAME_70B,
]


def main():
    for model_name in DEEPSEEK_MODEL_NAMES:
        report_output_path = RESULTS_PATH / f"inference-{model_name}-report.txt"
        report_content = ""

        inference_results_with_query, inference_results_wo_query = (
            _get_inference_output_content(model_name)
        )
        for query_usage in ["with-query-term", "without-query-term"]:
            inference_results = (
                inference_results_with_query
                if query_usage == "with-query-term"
                else inference_results_wo_query
            )

            if query_usage == "without-query-term":
                report_content += "\n\n"

            report_content += f"{'=' * REPORT_WIDTH}\n{model_name=}, {query_usage=}\n"
            report_content += _check_missing_reasoning_entry(inference_results)
            report_content += _calculate_scores_for_both_classes(inference_results)
            report_content += _check_tweets_with_wrong_classification(
                inference_results, query_usage
            )

        report_content += _get_n_tweets_with_wrong_classification_without_query_term(
            inference_results_with_query, inference_results_wo_query, 5
        )

        report_output_path.write_text(report_content, 'UTF-8')


def _get_inference_output_content(model_name):
    inference_results_wo_query = InferenceResults.model_validate_json(
        (
            RESULTS_PATH
            / "local_inference_output"
            / f"{model_name}-prompts-results-without-query-term.json"
        ).read_text()
    )
    inference_results_with_query = InferenceResults.model_validate_json(
        (
            RESULTS_PATH
            / "local_inference_output"
            / f"{model_name}-prompts-results-with-query-term.json"
        ).read_text()
    )
    return inference_results_with_query, inference_results_wo_query


def _check_missing_reasoning_entry(inference_results: InferenceResults) -> str:
    count = 0
    for result in inference_results.results:
        if result.reasoning == "...":
            count += 1
    return f"Entries without reasoning entry: {count}\n"


def _calculate_scores_for_both_classes(inference_results: InferenceResults) -> str:
    results = inference_results.results
    y_true = [result.ground_truth for result in results]
    y_pred = [result.sentiment for result in results]

    classification_report_ = classification_report(y_true, y_pred, digits=3)

    headers_macro = ["accuracy", "support"]
    table_macro = [[accuracy_score(y_true, y_pred), len(y_true)]]

    headers = ["class", "precision", "recall", "f1", "support"]
    table = [line.split() for line in classification_report_.split("\n")[2:-5]]

    return (
        f"\n{tabulate(table_macro, headers=headers_macro, tablefmt='simple_outline')}\n"
        f"{tabulate(table, headers=headers, tablefmt='simple_outline', floatfmt='.3f')}\n"
    )


def _check_tweets_with_wrong_classification(
    inference_results: InferenceResults, query_usage: str, n=5
) -> str:
    results = inference_results.results
    results = [result for result in results if result.ground_truth != result.sentiment]
    headers = ["ground_truth", "prediction", "tweet"]
    if query_usage == "with-query-term":
        headers.insert(0, "query_term")
        results = sorted(results, key=lambda x: x.query_term)  # type: ignore
        table = [
            [result.query_term, result.ground_truth, result.sentiment, result.tweet]
            for result in results
        ]
    else:
        table = [
            [result.ground_truth, result.sentiment, result.tweet] for result in results
        ]

    return (
        f"\nFirst {n} tweets with wrong classifications:\n"
        f"{tabulate(table[:n], headers=headers, tablefmt='simple_grid', maxcolwidths=[12] + [8] * (len(headers) - 2) + [70])}\n"
    )


def _get_n_tweets_with_wrong_classification_without_query_term(
    inference_results_with_query: InferenceResults,
    inference_results_without_query: InferenceResults,
    n: int,
) -> str:
    table = []
    k = 0
    for result_wo_query in inference_results_without_query.results:
        if k == n:
            break
        if result_wo_query.ground_truth != result_wo_query.sentiment:
            result_with_query = (
                _find_tweet_in_results_with_query_term_and_correct_class(
                    result_wo_query.tweet,
                    result_wo_query.sentiment,
                    inference_results_with_query,
                )
            )
            if result_with_query is not None:
                rows = [
                    [
                        result_wo_query.ground_truth,
                        result_wo_query.tweet,
                        "-",
                        result_wo_query.sentiment,
                        result_wo_query.reasoning,
                    ],
                    [
                        "",
                        "",
                        result_with_query.query_term,
                        result_with_query.sentiment,
                        result_with_query.reasoning,
                    ],
                ]
                table.extend(rows)
                k += 1

    headers = ["ground_truth", "tweet", "query_term", "prediction", "reasoning"]
    return (
        f"\n{k} tweets with wrong classification without query term but correct "
        f"classification with query term:\n"
        f"{tabulate(table, headers=headers, tablefmt='simple_grid', maxcolwidths=[8, 30, 8, 8, 40])}\n"
    )


def _find_tweet_in_results_with_query_term_and_correct_class(
    tweet: str, class_: str, inference_results: InferenceResults
) -> EnrichedModelOutput | None:
    for result in inference_results.results:
        if result.tweet == tweet:
            if result.sentiment != class_:
                return result
            return None
    return None


if __name__ == "__main__":
    main()
