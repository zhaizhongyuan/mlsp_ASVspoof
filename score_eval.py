from tDCF_python.evaluate_tDCF_asvspoof19 import evaluate_tDCF_asvspoof19
import argparse

if __name__ == "__main__":
    # Parser argument
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cm_score",
        required=False,
        type=str,
        default="./output/gmm_score.txt",
        help="path to dev/eval score file. For example, output/gmm_score.text",
    )
    parser.add_argument(
        "--asv_score",
        required=False,
        type=str,
        default="./tDCF_python/scores/ASVspoof2019_LA_eval_asv_scores.txt",
        help="path to asv score file (provided by ASVspoof organization). For example, tDCF_python/scores/ASVspoof2019_LA_eval_asv_scores.txt",
    )
    parser.add_argument(
        "--legacy",
        required=False,
        type=bool,
        default=True,
        help="legacy mode flag, if True, then use ASVspoof 2019's evaluation plan for t-DCF; if False, use ASVspoof 2021's evaluation plan",
    )
    args = parser.parse_args()

    # evaluate min t-dcf and eer
    evaluate_tDCF_asvspoof19(args.cm_score, args.asv_score, args.legacy)