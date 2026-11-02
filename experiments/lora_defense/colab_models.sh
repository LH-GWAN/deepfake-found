# Sourced by colab_run_budget.sh and colab_score_budget.sh, from the repository root: the
# buffalo_l pack (SCRFD and the w600k_r50 ArcFace the shield attacks) and inswapper, each
# checked against the digest the local runs used. Exports ARCFACE_ONNX.
BUFFALO=models/insightface/models/buffalo_l
if [ ! -f $BUFFALO/w600k_r50.onnx ]; then
    mkdir -p models/insightface/models
    wget -q -O models/insightface/models/buffalo_l.zip \
        https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip
    unzip -q -o models/insightface/models/buffalo_l.zip -d $BUFFALO
fi
echo "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43  $BUFFALO/w600k_r50.onnx" \
    | sha256sum -c -
export ARCFACE_ONNX=$PWD/$BUFFALO/w600k_r50.onnx
[ -f models/inswapper/inswapper_128.onnx ] || python scripts/fetch_inswapper.py
