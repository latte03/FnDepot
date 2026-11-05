#!/bin/bash
# HyAtlas 5.0.0-1 fpk 构建（同步上游 v4.9.1→v5.0.0：检索显式流水线 + MMR、别名进 store 层、catch-up 补抽、写路径单次 embedding + 倒排关键词索引、可逆 forget、/flush + 优雅停机、typed retire、链接事实与周期事件 series；fork 定制全保留，持久层换上游 1s flushLoop；详见 fnos-native/manifest 的 changelog 字段）
#
# 输入来源：
#   - fpk 控制层（cmd/config/wizard/manifest/ICON）：本目录归档原件
#   - Go 二进制：bin/hyatlas-go-linux-amd64（本分支源码 go1.26.5 本地编译，
#     CGO_ENABLED=1 -trimpath GOAMD64=v1，非上游 CI 产物；含 dashboard go:embed 内嵌前端，
#     故 dashboard 有改动必须重编后重新打包装入 SHA_BIN，只改 manifest 不算交付）
#   - 模型三件套（~355MB，超 GitHub 100MB 限制不入库）：按 MODEL_SRC → /tmp/b2-models
#     → 从 REF_FPK 内层 app.tgz 提取的顺序解析；sha256 断言防漂移。
#     历史沿革：REF_FPK 原指 4.2.5-1，后改指 4.3.3-1，两者均已从 download 目录消失；
#     再改指 4.5.0-2 产物（/vol2/1000/workspace/hy450-sync/out/，作者机器路径）——
#     该目录在本机已不存在，兜底路径再次断裂。现改为：REF_FPK 默认留空（= 显式声明
#     「本机无兜底 fpk」），模型一律走 MODEL_SRC；本机现役模型在
#     /vol1/@appcenter/hyatlas/models，三件套 sha 与下方 SHA_ONNX/SHA_ORT/SHA_TOK
#     逐件相符（2026-11 实测）。缺失时脚本直接 FATAL 并点名 MODEL_SRC，不再报错到
#     一个不存在的 fpk 路径上。
#
# 用法: bash build-fpk.sh [输出路径]
# ⚠ OUT 默认值是 /vol02/1000-1-13b246aa/… = fuse.rclone WebDAV 挂载（见 mount），
#   脚本要写 out+'.tmp' 再 os.replace，走该挂载有大文件重命名风险；
#   打包请显式传本地路径：bash build-fpk.sh /vol2/1000/download/hyatlas-4.6.0-1-x86.fpk
#
# 与 4.1.1-2 时代的字节级等价构建差异：
#   - 4.1.1-5 控制层与二进制均有变更，不再做 REF 产物 sha 等价断言；
#     末尾改为结构校验（manifest 版本、shares 自愈回调在位、app payload 完整）
#   - fnpack build 校验步骤省略（fnpack 重压 app.tgz 会改字节）
#   - manifest checksum 占位符替换发生在 tar 组装之后，对产物无影响，此处保留
#     占位符原件直拼
#
# ⚠ 外层 tar 身份已**显式钉死**为 905/899 Harness/Harness（= 5.0.0-1 出货件实测值）。
#   原先取 st_uid/st_gid + pwd/grp 反查，等于把「跑脚本的人」写进产物身份：同样的内容
#   换个人构建 sha 就变（实测 agan uid1000/gid1001 → 243,464,853B；钉死后 905/899 →
#   243,464,854B = 官方 sha256 4ee0a695…）。需要别的身份时用 FPK_UID / FPK_GID /
#   FPK_UNAME / FPK_GNAME 覆盖，改这四个值会直接改变产物 sha256。
#
# ── 复现判据（5.0.0-1 冻结基线，2026-11 实测）──────────────────────────────
#   未修改树（git 6ed0ae0）+ 二进制用 golang:1.27.1-bookworm 重编（= bin/ 的 bb3376ba…）
#   + MODEL_SRC=/vol1/@appcenter/hyatlas/models + 外层身份 905/899 Harness/Harness：
#     产物 = 243,464,854 B
#     sha256 = 4ee0a69595b1f621e842d5aacc4f61680f558c62515b230219fba1a02f744835
#     = 官方 5.0.0-1 原件，cmp 逐字节相同；连打两次 sha 相同（确定性）。
#   工具链必须带 -bookworm 后缀：golang:1.27 已浮动到 go1.27.3/trixie，golang:1.27.1 的
#   默认变体也是 trixie(gcc 14)，两者分别产出 16,898,208 / 16,873,448 B，只有
#   golang:1.27.1-bookworm 给出 16,873,936 B。判别式：二进制 .comment 段
#   `GCC: (Debian 12.2.0-14+deb12u1) 12.2.0`。
#   判据用法：打 5.0.0-N 时，与 5.0.0-1 的差异应**只来自本轮代码/控制层改动**；若大小、
#   成员清单或元数据出现计划外偏移，先查本节前提（工具链/模型 sha/外层身份）是否被破坏。
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${1:-/vol2/1000/download/hyatlas-5.0.0-1-x86.fpk}"
VER="5.0.0-1"   # 本包版本，manifest 校验与身份闸提示共用（上游版本 + fork 修订号）
REF_FPK="${REF_FPK:-}"   # 本机无兜底 fpk（作者机器的 /vol2/1000/workspace/hy450-sync/ 已不存在）→ 默认留空，模型走 MODEL_SRC；需要时显式传 REF_FPK=...

# 4.9.0-1 是**上游同步版**（v4.7.3→v4.9.0，Go 源码与前端 embed 均有改动），本轮**重编**，
# 不允许复用旧二进制：REUSE_BIN 默认置 0（要复用必须显式 REUSE_BIN=1 且那是打包层修订）。
# 5.0.0-1 重编注意：go build 必须带 -buildvcs=false，否则 git 仓库内构建会盖 VCS 戳，
# 两份不同挂载路径的产物因 .git 有无不一致（4.9.1-1 出货件 buildinfo 无 vcs 键，与本闸对齐）。
# SHA_BIN 取自出包前置门禁 rebuild-twice.sh 的权威值：/vol2 与 /vol1 两份不同挂载路径
# 重编逐字节一致（go1.26.5，CGO_ENABLED=1，-trimpath，GOAMD64=v1）。
SHA_BIN="bb3376bab085b4a568c48ad9ce2bd15e1c69c96509cea66cac2bc371e5459ae3"   # 实测：v5.0.0 合并树 + fork 定制保留(-trimpath -buildvcs=false,无 -s -w)；16,873,936B
REUSE_BIN="${REUSE_BIN:-0}"   # 1 = 本轮为打包层修订，允许复用 bin/ 现有二进制（非重编）；重编发布时置 0
SHA_ONNX="8a3f371a7e535e25d3d5a0ff0c0501a605ef0b62577800d2bf4b1fc76d6cbcf1"
SHA_ORT="99458e9d185dfa1a9b5f6510790ede3bedc25dea378adb904ce292b517eeaecf"
SHA_TOK="7dfbf1966ebf99d471c3796e9b457329d2b2182b817e144f1e904b957745c839"

# ── SHA_BIN 自检闸（防"只改 manifest 不动二进制"）────────────────────────────
# sha256sum -c 只能证明"bin/ 里的文件 == SHA_BIN 写的值"，两者一起写旧值照样全绿。
# 所以在组装前硬性拒绝：仍是占位符 / 不是 64 位十六进制 / 等于历史已出货二进制的 sha。
# ⚠️ REUSE_BIN=1 只在"纯打包层修订（Go 源码与模型零变更）"时用：此时身份闸（下面 case）
#    降级为 WARN 而非 FATAL；但 SHA_BIN == bin/ 实际 sha 的强校验**始终生效**，
#    不会放过"脚本写旧值"的情形。本轮 4.9.0-1 是上游同步版，REUSE_BIN=0（重编）。
case "$SHA_BIN" in
  TBD*|*recompute*|*[!0-9a-fA-F]*)
    echo "FATAL: SHA_BIN 非法（占位符或不是 64 位十六进制 sha256）: '$SHA_BIN'" >&2
    echo "       对 bin/hyatlas-go-linux-amd64 跑 sha256sum 后填入实际值" >&2; exit 1;;
esac
[ "${#SHA_BIN}" = 64 ] || { echo "FATAL: SHA_BIN 长度 ${#SHA_BIN} != 64" >&2; exit 1; }
[ -f "$HERE/bin/hyatlas-go-linux-amd64" ] || { echo "FATAL: 缺 $HERE/bin/hyatlas-go-linux-amd64" >&2; exit 1; }
ACTUAL_BIN_SHA="$(sha256sum "$HERE/bin/hyatlas-go-linux-amd64" | cut -d' ' -f1)"
[ "$ACTUAL_BIN_SHA" = "$SHA_BIN" ] || { echo "FATAL: SHA_BIN($SHA_BIN) != bin/ 实际 sha($ACTUAL_BIN_SHA)" >&2; exit 1; }
GATE_FATAL=1
[ "$REUSE_BIN" = "1" ] && GATE_FATAL=0
_identity_gate() {
  if [ "$GATE_FATAL" = "1" ]; then echo "FATAL: $1" >&2; exit 1; else echo "WARN(REUSE_BIN=1 打包层修订): $1" >&2; fi
}
case "$ACTUAL_BIN_SHA" in
  4a39e8e02cf564c1b0c085a1ec216d02ba0eb4bea952ca95b77c1163bd76e95b)
    _identity_gate "bin/ 仍是 4.5.0-2 出货二进制（4a39e8e0），不能以 $VER 名义打包";;
  01bec1798c50456672b2a2980aa931b994d10e55f6c97caebea60b3ce73abdce)
    _identity_gate "bin/ 仍是 4.3.3-1 出货二进制（01bec179），不能以 $VER 名义打包";;
  83740365a1235a2dc0086be35c1d7667dd7eb7817c0ff09d1b229e3df18d2b4d)
    _identity_gate "bin/ 仍是 4.3.0-1 出货二进制（83740365），不能以 $VER 名义打包";;
  f9acdf9f4917df1f73a609ce975d4651684a6d0ccdd533f071a7ee66a8c47c2d)
    echo "WARN: bin/ 是 dashboard 汉化修复前的 f9acdf9f —— 若 i18n 补全已合入则本值应已变化" >&2;;
  510b83cc2f9d7b9d670c0b523df3a343cd81077a0e70a514618b13133c159b34)
    _identity_gate "bin/ 仍是 4.7.2-1 出货二进制（510b83cc），不能以 $VER 名义打包";;
esac
echo "SHA_BIN 自检通过: $ACTUAL_BIN_SHA ($(stat -c '%s' "$HERE/bin/hyatlas-go-linux-amd64") B, REUSE_BIN=$REUSE_BIN)"

STAGING="$(mktemp -d)"
trap 'rm -rf "$STAGING"' EXIT

echo "=== 1/4 app payload ==="
APPROOT="$STAGING/app_new"
mkdir -p "$APPROOT/gateway" "$APPROOT/models" "$APPROOT/ui/images"
cp "$HERE/bin/hyatlas-go-linux-amd64" "$APPROOT/hyatlas-go"
cp "$HERE/gateway/gateway_proxy.py" "$APPROOT/gateway/"
cp "$HERE/config/env.example" "$APPROOT/env.example"
cp "$HERE/ui/config" "$APPROOT/ui/"
cp -a "$HERE/ui/images/." "$APPROOT/ui/images/"

# 模型三件套：MODEL_SRC → /tmp/b2-models → 现有 fpk 提取
resolve_models() {
  local cand
  for cand in "${MODEL_SRC:-}" /tmp/b2-models; do
    if [ -n "$cand" ] && [ -f "$cand/model_int8.onnx" ] && [ -f "$cand/libonnxruntime.so" ] && [ -f "$cand/tokenizer.json" ]; then
      echo "$cand"; return 0
    fi
  done
  return 1
}
MODEL_DIR=""
if MODEL_DIR="$(resolve_models)"; then
  echo "models from: $MODEL_DIR"
  cp "$MODEL_DIR/libonnxruntime.so" "$MODEL_DIR/model_int8.onnx" "$MODEL_DIR/tokenizer.json" "$APPROOT/models/"
else
  echo "models not found on disk, extracting from ${REF_FPK:-<未设置>} (inner app.tgz)"
  [ -f "${REF_FPK:-}" ] || { cat >&2 <<EOM
FATAL: 模型三件套找不到，无法继续打包。
  已尝试的查找顺序：MODEL_SRC → /tmp/b2-models → REF_FPK 内层 app.tgz
  当前 MODEL_SRC=${MODEL_SRC:-<未设置>}，REF_FPK=${REF_FPK:-<未设置>}

  说明：这台机器上没有 /vol2/1000/workspace/hy450-sync/（那是作者机器的路径，
  4.5.0-2 产物早已不在），所以 REF_FPK 兜底默认留空——这不是配置写错，是「本机没有兜底 fpk」。

  本机正解：显式指定现役模型目录
      MODEL_SRC=/vol1/@appcenter/hyatlas/models bash build-fpk.sh <输出路径>
  该目录三件套 sha 与脚本里的 SHA_ONNX/SHA_ORT/SHA_TOK 逐件相符（2026-11 实测）。
  确需用 fpk 兜底时，再显式传 REF_FPK=/path/to/xxx.fpk。
EOM
  exit 1; }
  tar -xf "$REF_FPK" -C "$STAGING" app.tgz
  mkdir -p "$STAGING/mx" "$APPROOT/models"
  tar -xzf "$STAGING/app.tgz" -C "$STAGING/mx" ./models
  cp "$STAGING/mx/models/libonnxruntime.so" "$STAGING/mx/models/model_int8.onnx" "$STAGING/mx/models/tokenizer.json" "$APPROOT/models/"
fi
chmod 755 "$APPROOT/hyatlas-go" "$APPROOT/models/libonnxruntime.so"

# 输入完整性断言
echo "$SHA_BIN  $APPROOT/hyatlas-go"     | sha256sum -c --quiet
echo "$SHA_ONNX  $APPROOT/models/model_int8.onnx" | sha256sum -c --quiet
echo "$SHA_ORT  $APPROOT/models/libonnxruntime.so"| sha256sum -c --quiet
echo "$SHA_TOK  $APPROOT/models/tokenizer.json"   | sha256sum -c --quiet

echo "=== 2/4 deterministic app.tgz (gzip mtime=0, 内层元数据钉死) ==="
cd "$APPROOT"
python3 - << 'PYEOF'
import gzip, os, stat, tarfile
# 元数据钉死自已发布产物 app.tgz（内层 tar，uid/gid=0，tar 路径带 ./ 前缀）
INNER_PINS = {
    './app.tgz':                  {'mode': 0o644, 'mtime': 1788665563},
    './gateway':                  {'mode': 0o755, 'mtime': 1788777600},
    './gateway/gateway_proxy.py': {'mode': 0o644, 'mtime': 1788777600},
    './hyatlas-go':               {'mode': 0o755, 'mtime': 1788777600},
    './models':                   {'mode': 0o755, 'mtime': 1788665563},
    './models/libonnxruntime.so': {'mode': 0o755, 'mtime': 1788777600},
    './models/model_int8.onnx':   {'mode': 0o644, 'mtime': 1788665563},
    './models/tokenizer.json':    {'mode': 0o644, 'mtime': 1788665563},
    './ui':                       {'mode': 0o755, 'mtime': 1788777600},
    './ui/config':                {'mode': 0o755, 'mtime': 1788777600},
    './ui/images':                {'mode': 0o755, 'mtime': 1788777600},
    './ui/images/icon-256.png':   {'mode': 0o755, 'mtime': 1788777600},
    './ui/images/icon-64.png':    {'mode': 0o755, 'mtime': 1788777600},
    './env.example':              {'mode': 0o644, 'mtime': 1788777600},
}
raw = gzip.GzipFile(filename='', mode='wb', compresslevel=6, mtime=0, fileobj=open('app.tgz','wb'))
with tarfile.open(fileobj=raw, mode='w') as tf:
    for name in sorted(INNER_PINS, key=str.lower):
        rel = name[2:]
        st = os.lstat(rel)
        ti = tarfile.TarInfo(name)
        ti.size = st.st_size; ti.mtime = INNER_PINS[name]['mtime']; ti.mode = INNER_PINS[name]['mode']
        ti.uid = 0; ti.gid = 0; ti.uname = ''; ti.gname = ''
        if stat.S_ISDIR(st.st_mode):
            ti.type = tarfile.DIRTYPE; ti.size = 0; tf.addfile(ti)
        else:
            ti.type = tarfile.REGTYPE
            with open(rel,'rb') as f: tf.addfile(ti, f)
raw.close()
print('app.tgz built deterministically')
PYEOF
mv app.tgz "$STAGING/app.tgz"

echo "=== 3/4 deterministic fpk assemble (外层元数据钉死) ==="
cd "$STAGING"
# 控制层：仓库归档原件直取（字节已验证 == fpk 内原件）
cp -a "$HERE/cmd" "$HERE/config" "$HERE/wizard" .
cp -a "$HERE/ICON.PNG" "$HERE/ICON_256.PNG" "$HERE/manifest" .
# ⚠️ .bak 闸门（与 emby-in-one/build.sh 同款）：$STAGING 是即将拼外层 fpk 的源目录（当前 cwd）。
#    上面 `cp -a cmd config wizard` 会把源码目录里的 *.bak_* 一并拷进来；下方 python 按 os.listdir
#    逐个取 OUTER_PINS['cmd/<child>']，混入的备份会直接 KeyError 崩，这里提前给一句人话 FATAL。
#    （内层 app.tgz 走 INNER_PINS 白名单+定档 app 源，本就不含备份，无需重复查。）
BAK="$(find "$STAGING" -name '*.bak*' -o -name '*_~' -o -name '*.orig' 2>/dev/null | head -20)"
if [ -n "$BAK" ]; then
  echo "FATAL: 包内含备份文件，先移出再打包：" >&2
  echo "$BAK" >&2
  exit 1
fi
python3 - "$OUT" << 'PYEOF'
import gzip, os, stat, sys, tarfile
out = sys.argv[1]
# 外层钉死自已发布产物（uid/gid=hermes-studio 运行身份，tar 路径无 ./ 前缀）
OUTER_PINS = {
    'app.tgz':    {'mode': 0o644, 'mtime': 1788777601},
    'cmd':        {'mode': 0o755, 'mtime': 1788777602},
    'cmd/config_callback':    {'mode': 0o755, 'mtime': 1788777602},
    'cmd/config_init':        {'mode': 0o755, 'mtime': 1788777602},
    'cmd/install_callback':   {'mode': 0o755, 'mtime': 1788777602},
    'cmd/install_init':       {'mode': 0o755, 'mtime': 1788777602},
    'cmd/main':               {'mode': 0o755, 'mtime': 1788777602},
    'cmd/uninstall_callback': {'mode': 0o755, 'mtime': 1788777602},
    'cmd/uninstall_init':     {'mode': 0o755, 'mtime': 1788777602},
    'cmd/upgrade_callback':   {'mode': 0o755, 'mtime': 1788777602},
    'cmd/upgrade_init':       {'mode': 0o755, 'mtime': 1788777602},
    'config':                 {'mode': 0o755, 'mtime': 1788777602},
    'config/env.example':     {'mode': 0o644, 'mtime': 1788777602},
    'config/privilege':       {'mode': 0o644, 'mtime': 1788777602},
    'config/resource':        {'mode': 0o644, 'mtime': 1788777602},
    'ICON.PNG':               {'mode': 0o705, 'mtime': 1788777601},
    'ICON_256.PNG':           {'mode': 0o705, 'mtime': 1788777601},
    'manifest':               {'mode': 0o644, 'mtime': 1788777601},
    'wizard':                 {'mode': 0o755, 'mtime': 1788777602},
    'wizard/install':         {'mode': 0o644, 'mtime': 1788777602},
    'wizard/uninstall':       {'mode': 0o644, 'mtime': 1788777602},
}
# 外层 tar 身份显式钉死：905/899 Harness/Harness = 5.0.0-1 出货件实测值
# （官方原件 243,464,854B / sha256 4ee0a69595b1f621e842d5aacc4f61680f558c62515b230219fba1a02f744835
#   在「二进制 go1.27.1-bookworm 重编 = bb3376ba + 模型三件套 sha 逐件相符」前提下可逐字节复现）。
# 原实现是 st_uid/st_gid + pwd/grp 反查 ⇒ 构建用户成为产物身份的一部分：同一份内容
# 换个人跑（uid 1000/1001 agan）就得到 243,464,853B，与内容无关却让复现失真。
# 需要别的身份时用 FPK_UID / FPK_GID / FPK_UNAME / FPK_GNAME 覆盖；改这四个值会直接改变产物 sha256。
OUTER_UID   = int(os.environ.get('FPK_UID', 905))
OUTER_GID   = int(os.environ.get('FPK_GID', 899))
OUTER_UNAME = os.environ.get('FPK_UNAME', 'Harness')
OUTER_GNAME = os.environ.get('FPK_GNAME', 'Harness')

class BareDirInfo(tarfile.TarInfo):
    # 原打包脚本同款：目录条目名不带尾斜杠
    def get_info(self):
        info = tarfile.TarInfo.get_info(self)
        if info['type'] == tarfile.DIRTYPE and info['name'].endswith('/'):
            info['name'] = info['name'].rstrip('/')
        return info

def ti_for(path, name, pins):
    st = os.lstat(path)
    ti = BareDirInfo(name)
    ti.size = st.st_size; ti.mtime = pins['mtime']; ti.mode = pins['mode']
    ti.uid, ti.gid, ti.uname, ti.gname = OUTER_UID, OUTER_GID, OUTER_UNAME, OUTER_GNAME
    if stat.S_ISDIR(st.st_mode): ti.type = tarfile.DIRTYPE; ti.size = 0
    elif stat.S_ISLNK(st.st_mode): ti.type = tarfile.SYMTYPE; ti.linkname = os.readlink(path)
    else: ti.type = tarfile.REGTYPE
    return ti

raw = gzip.GzipFile(filename='', mode='wb', compresslevel=6, mtime=0, fileobj=open(out+'.tmp','wb'))
with tarfile.open(fileobj=raw, mode='w') as tf:
    for e in ['app.tgz'] + sorted(['cmd', 'config', 'ICON.PNG', 'ICON_256.PNG', 'manifest', 'wizard'], key=str.lower):
        pins = OUTER_PINS[e]
        if os.path.isdir(e) and not os.path.islink(e):
            tf.addfile(ti_for(e, e, pins))
            for child in sorted(os.listdir(e)):
                cp = os.path.join(e, child)
                of = open(cp,'rb') if os.path.isfile(cp) else None
                tf.addfile(ti_for(cp, f'{e}/{child}', OUTER_PINS[f'{e}/{child}']), of)
                if of: of.close()
        else:
            of = open(e,'rb')
            tf.addfile(ti_for(e, e, pins), of)
            of.close()
raw.close()
os.replace(out + '.tmp', out)
print('fpk assembled')
PYEOF

echo "=== 4/4 verify (structural) ==="
# ⚠ 本段整体关掉 pipefail：校验全是 `tar -xOf "$OUT" <成员> | grep -q PATTERN` 形态。
#   grep -q 命中即退出并关闭读端 → tar 继续写第二块时吃 SIGPIPE(141) → 在 pipefail 下
#   管道状态取 141，`if !` 就把「命中」误判成「缺失」。cmd/main(16003B) 与
#   install_callback(11036B) 都超过 tar 的 10240B 记录长，所以只有它们会踩：
#   同一份字节连跑 12 次实测 9 次 FAIL / 3 次 PASS（内容用 grep -c 复核恒为 2）。
#   关掉 pipefail 后管道取末段状态，语义不变：tar 失败 → 无输出 → grep 不命中 → 仍然 FAIL。
#   注意：本段在 fpk 写完之后，改这里**不会**改变产物字节。
set +o pipefail
GOT_SHA=$(sha256sum "$OUT" | cut -d' ' -f1)
GOT_SIZE=$(stat -c '%s' "$OUT")
FAIL=0

# 1) gzip+tar 可解
if ! tar -tzf "$OUT" >/dev/null 2>&1; then
  echo "FAIL: fpk is not a valid gzip+tar" >&2; FAIL=1
fi

# 2) manifest 版本/appname
GOT_VER=$(tar -xOf "$OUT" manifest | awk -F'= *' '$1 ~ /^version/ {print $2}' | tr -d ' ')
GOT_APP=$(tar -xOf "$OUT" manifest | awk -F'= *' '$1 ~ /^appname/ {print $2}' | tr -d ' ')
if [ "$GOT_VER" != "$VER" ] || [ "$GOT_APP" != "hyatlas" ]; then
  echo "FAIL: manifest appname=$GOT_APP version=$GOT_VER (expected hyatlas $VER)" >&2; FAIL=1
fi

# 3) shares 校验/自愈回调在位（install/upgrade 双回调都含特征串）
for f in install_callback upgrade_callback; do
  if ! tar -xOf "$OUT" "cmd/$f" 2>/dev/null | grep -q 'shares is NOT a symlink'; then
    echo "FAIL: cmd/$f missing shares-heal/verify logic" >&2; FAIL=1
  fi
done

# 3b) 四脚本各带一份 resolve_share_root（禁止 source 兄弟文件）
for f in main install_callback upgrade_callback uninstall_callback; do
  if ! tar -xOf "$OUT" "cmd/$f" 2>/dev/null | grep -q 'resolve_share_root'; then
    echo "FAIL: cmd/$f missing resolve_share_root" >&2; FAIL=1
  fi
done

# 3c) 卸载双选：wizard/uninstall 存在 + 字段名 wizard_delete_data 两处齐
if ! tar -xOf "$OUT" wizard/uninstall 2>/dev/null | grep -q 'wizard_delete_data'; then
  echo "FAIL: wizard/uninstall missing wizard_delete_data" >&2; FAIL=1
fi
if ! tar -xOf "$OUT" cmd/uninstall_callback 2>/dev/null | grep -q 'wizard_delete_data'; then
  echo "FAIL: cmd/uninstall_callback does not read wizard_delete_data" >&2; FAIL=1
fi

# 3d) 已证伪的路径写法不得出现（dirname TRIM_DATA_SHARE_PATHS / ${APP_DIR}/shares / 单数 /share/）
#     ⚠️ 必须先剥掉 `#` 注释：脚本里保留的“旧写法为何被证伪”说明注释本身就含这两个字符串，
#     不剥注释会把反面教材误判成代码回退，让构建被自己的注释拦下。
for f in main install_callback upgrade_callback uninstall_callback; do
  if tar -xOf "$OUT" "cmd/$f" 2>/dev/null | sed 's/#.*$//' | grep -qE 'dirname.*TRIM_DATA_SHARE_PATHS|\$\{APP_DIR\}/shares'; then
    echo "FAIL: cmd/$f still uses disproved share path form" >&2; FAIL=1
  fi
done

# 4) app.tgz payload 完整（二进制 + 模型三件套 + 网关 + ui/config）
tar -xf "$OUT" -O app.tgz > "$STAGING/inner.tgz"
for p in './hyatlas-go' './models/model_int8.onnx' './models/libonnxruntime.so' './models/tokenizer.json' './gateway/gateway_proxy.py' './ui/config' './env.example'; do
  if ! tar -tzf "$STAGING/inner.tgz" "$p" >/dev/null 2>&1; then
    echo "FAIL: app.tgz missing $p" >&2; FAIL=1
  fi
done

# 5) 内层二进制 = 本次构建的新二进制（防旧包混入）
INNER_BIN_SHA=$(tar -xzf "$STAGING/inner.tgz" -O ./hyatlas-go | sha256sum | cut -d' ' -f1)
if [ "$INNER_BIN_SHA" != "$SHA_BIN" ]; then
  echo "FAIL: inner hyatlas-go sha mismatch ($INNER_BIN_SHA != $SHA_BIN)" >&2; FAIL=1
fi

if [ "$FAIL" != "0" ]; then
  echo "STRUCTURE CHECK FAILED" >&2
  exit 1
fi
set -o pipefail
echo "PASS: structure ok (manifest $VER, share two-hop resolve + uninstall wizard present, app payload complete incl env.example)"
echo "$OUT  $((GOT_SIZE/1024/1024))MB  sha256=$GOT_SHA"
