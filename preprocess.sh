echo "extracting features: lfcc"
echo "preparing training data"
python3 data_processing.py --data_path ../LA/ASVspoof2019_LA_train/flac \
                        --output_path ./data/train/ \
                        --label_path ../LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt \
                        --ftype lfcc

echo "preparing dev data"
python3 data_processing.py --data_path ../LA/ASVspoof2019_LA_dev/flac \
                        --output_path ./data/dev/ \
                        --label_path ../LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt \
                        --ftype lfcc

echo "preparing test data"
python3 data_processing.py --data_path ../LA/ASVspoof2019_LA_eval/flac \
                        --output_path ./data/test/ \
                        --label_path ../LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt \
                        --ftype lfcc