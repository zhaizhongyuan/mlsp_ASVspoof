echo "testing with dev data on features: mfcc"
python3 gmm_test.py --data_path ./data/dev/mfcc.pkl \
                    --model_path_bon ./model/mfcc/bon.gmm \
                    --model_path_sp ./model/mfcc/sp.gmm

echo "testing with test data on features: mfcc"
python3 gmm_test.py --data_path ./data/test/mfcc.pkl \
                    --model_path_bon ./model/mfcc/bon.gmm \
                    --model_path_sp ./model/mfcc/sp.gmm