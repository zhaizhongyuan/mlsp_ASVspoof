echo "extracting features: mfcc"
echo "preparing training data"
python3 data_processing.py --data_path /mnt/LA/ASVspoof2019_LA_train/flac \
                        --output_path ./data/train/ \
                        --label_path /mnt/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt \
                        --ftype mfcc

echo "preparing dev data"
python3 data_processing.py --data_path /mnt/LA/ASVspoof2019_LA_dev/flac \
                        --output_path ./data/dev/ \
                        --label_path /mnt/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt \
                        --ftype mfcc

echo "preparing test data"
python3 data_processing.py --data_path /mnt/LA/ASVspoof2019_LA_eval/flac \
                        --output_path ./data/test/ \
                        --label_path /mnt/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt \
                        --ftype mfcc