# Mac 端脚本的公共设置（被其他脚本 source）。
# 用户给的别名是 s01_remote，但这台 Mac 的 ~/.ssh/config 里只有 s01-remote（Tailscale）和 s01（局域网），
# 两者已核实指向 spark-b935。脚本不修改 SSH 配置；如需改用其他别名，设置 SPARK_HOST 即可。
SPARK_HOST="${SPARK_HOST:-s01-remote}"
SPARK_REPO="${SPARK_REPO:-research/repos/omni-serving-research}"   # 相对 Spark 上的 $HOME
spark() { ssh -o BatchMode=yes -o ConnectTimeout=15 "$SPARK_HOST" "$@"; }
