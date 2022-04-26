# For documentation, please refer to "doc/tasks.md"

dataDir="/data/public_data/"  # path/to/your/data


scale=0.5

python streamyolo_det.py \
	--data-root "$dataDir/Argoverse-1.1/tracking" \
	--annot-path "$dataDir/Argoverse-HD/annotations/val.json" \
	--fps 30 \
	--weights "/data/output/one_x/m_s50_dfp_flip_tal_04_17/best_ckpt.pth" \
	--in_scale 0.5 \
	--no-mask \
	--out-dir "/data/online_resuklt/m_s50" \
	--overwrite \
	--config "/data/worker/StreamYOLO/cfgs/m_s50_onex_dfp_tal_flip.py" \
   &&
python streaming_eval.py \
	--data-root "$dataDir/Argoverse-1.1/tracking" \
	--annot-path "$dataDir/Argoverse-HD/annotations/val.json" \
	--fps 30 \
	--eta 0 \
	--result-dir "/data/online_resuklt/m_s50" \
	--out-dir "/data/online_resuklt/m_s50" \
	# --vis-dir "/data/online_resuklt/m_s50/vis" \
	# --vis-scale 0.5 \
	#--overwrite \
