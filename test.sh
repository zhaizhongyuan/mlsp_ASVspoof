echo "testing with dev data on features: lfcc"
python3 gmm_test.py --data_path ./data/dev/lfcc.pkl \
                    --model_path_bon ./model/lfcc/bon_fit_all_partial.gmm \
                    --model_path_sp ./model/lfcc/spoof_fit_all_partial.gmm

echo "testing with test data on features: lfcc"
python3 gmm_test.py --data_path ./data/test/lfcc.pkl \
                    --model_path_bon ./model/lfcc/bon_fit_all_partial.gmm \
                    --model_path_sp ./model/lfcc/spoof_fit_all_partial.gmm