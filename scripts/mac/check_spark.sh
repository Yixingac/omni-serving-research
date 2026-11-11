#!/usr/bin/env bash
# Mac 端：检查能否以非交互方式连上 Spark，并打印 Spark 的环境摘要。
#   scripts/mac/check_spark.sh
set -uo pipefail
. "$(dirname -- "${BASH_SOURCE[0]}")/spark_common.sh"
if ! spark true 2>/dev/null; then
  echo "无法以 BatchMode 连接 $SPARK_HOST（需要交互认证、网络不通或别名不存在）。"
  exit 1
fi
spark "bash -s" < "$(dirname -- "${BASH_SOURCE[0]}")/../spark/check_environment.sh"
