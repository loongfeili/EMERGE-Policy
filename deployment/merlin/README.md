# Merlin 上的可扩展 RoboDojo / EMERGE 评测

2026-10-04 的 AIDP Responses 扩容支持共享 API 配额协调器。推理配置可设置
`api_limiter: {"rpm": 90, "port": 8100}`，只在该推理任务 rank 0 启动；
评测配置设置 `require_api_limiter: true`，双方通过私有 env_map 注入同一
`EMERGE_RATE_LIMIT_TOKEN`。它仅签发配额，不接收模型 API key、图像或提示。
评测节点从服务清单获取 `api_limiter_url`，每次 Responses 请求（含重试）先取得配额；
协调器故障时禁止绕过限速。429 会使所有节点暂停至少 65 秒并将速率降低 25%。
`services/<推理组>/api-quota.json` 记录速率、排队数及限流事件。

配额等待不计入子代理和视觉验证的计算超时；模型调用、工具执行仍受原有超时约束。
评测配置可显式设置 `episode_timeout_s` 来容纳排队产生的墙钟延迟，必须在报告中记录；
它不改变 40 轮 Agent 上限或 RoboDojo 仿真步数。64 卡限速运行使用每卡 1 worker，
并非旧 `full64.json` 的每卡 2 worker。各子服务池合并前必须校验源码版本、副本数和健康状态。

本目录是部署入口。算法代码以 `https://github.com/loongfeili/EMERGE-Policy` 的 `robodojo` 分支为发布源；环境以 `https://github.com/loongfeili/RoboDojo` 为源。每次发布固定完整 commit，每个 Pod 都从 GitHub fetch 并 checkout 该 commit（隔离全局Git配置，HTTP/1.1传输，最多5次有界重试）；不把未提交补丁叠在部署源码上。

本流程已完成独立冷启动验收：2节点×1 L20、4条正式episode、0基础设施错误，两节点退出码0，最终覆盖完整且无重复；官方判分、轨迹与12路视频完整解码一致。该验证用于确认工程链路，不能代表全量benchmark或32/64卡吞吐。

已验证发布为 `merlin-v1-20260928-r5`，运行代码固定 `9c96284a27f8e445688694dd955024436eeb80a5`，发布清单SHA256为 `1b31067b39fdf13f5859d7ecd080c3c9355bb3ee9b1d2817ecc151a83d7cf2ab`。最终收尾提交仅更新本文，未改变运行代码和封存release；release内README保留发布时快照，以分支本文及下述HDFS RUNBOOK为最终交接文档。

- [L20验证job](https://seed.bytedance.net/development/instance/jobs/99402708e9af4f6d?trialId=422838308)：trial `422838308`，平台终态 `done`，两个Pod退出码0。
- [推理job](https://seed.bytedance.net/development/instance/jobs/a11aeaf35a66d443?trialId=422837793)：trial `422837793`，名称 `geometry_seg_infer`，验收后主动停止。
- 验收及流程目录：`/mnt/hdfs/__MERLIN_USER_DIR__/emerge_robodojo_20260925/diagnostics/merlin-standard-20260928-r5/`，包含 `RUNBOOK.md`、`REPORT.md`、`final-verification.json` 和 `final-trace-video-audit.json`。
- 结果路径：`<结果根>/merlin-v1-20260928-r5/runs/merlin-v1-20260928-r5-verify-seed0/`；历史服务注册路径：`<结果根>/merlin-v1-20260928-r5/services/`。服务已停止，不能继续使用历史地址；新实验创建新release和推理任务。

|工程验证episode|官方成功|官方得分|结束原因|
|---|---|---:|---|
|stack_bowls layout0 seed0|是|1.00|成功|
|stack_bowls layout1 seed0|否|0.15|40轮Agent预算耗尽|
|build_tower layout0 seed0|否|0.10|40轮Agent预算耗尽|
|build_tower layout1 seed0|否|0.00|40轮Agent预算耗尽|

实际成功1/4。四条轨迹记录到203次π推理、7条VGGT耗时记录、14条SAM耗时记录和39次视觉监控；存在内置skill读取，没有WAM调用，96个文本产物扫描未发现本次API key。54项pytest和6项部署单元测试通过；8/32/64卡L20及8卡推理配置dry-run通过，未提交全量实验。

## 固定位置与版本

|用途|开发机位置|Merlin Pod 位置|
|---|---|---|
|EMERGE开发仓库|`/mnt/bn/ic-vlm/personal/loongfei.li03/AgenticVLA08/EMERGE-Policy`|`/home/tiger/EMERGE-Policy`|
|RoboDojo开发仓库|`/mnt/bn/ic-vlm/personal/loongfei.li03/AgenticVLA08/RoboDojo`|`/home/tiger/RoboDojo`|
|资产/运行时/模型/发布清单|`/mnt/hdfs/__MERLIN_USER_DIR__/emerge_robodojo_20260925`|`/mnt/hdfs/emerge_cache`（只读）|
|结果根|`/mnt/hdfs/_BYTE_DATA_SEED_/ssd_hldy/user/lilongfei.xjgm/emerge_merlin`|`/mnt/hdfs/emerge_output/emerge_merlin`（读写）|
|本机私有环境参数|`/home/tiger/.config/emerge-robodojo/merlin-env.json`|平台 `job_config.job_template_config.env_map`|
|本机密钥说明文档|`/home/tiger/.config/emerge-robodojo/MERLIN_CREDENTIALS.md`|不复制此文档；启动后生成本机0600 agent配置|

缓存真实HDFS地址：`hdfs://harunawl/home/byte_data_seed_wl/user/loongfei.li03/emerge_robodojo_20260925`。
结果真实HDFS地址：`hdfs://haruna/home/byte_data_seed/ssd_hldy/user/lilongfei.xjgm/emerge_merlin`。
两种挂载路径不是同一个 namespace，不要互相替换前缀猜路径。

缓存下 `assets/assets.tar.zst` 是约36 GiB的资产；`runtime/` 是环境、解释器、图形兼容层和依赖源码；`models/manifest.json` 固定π0.5/VGGT/SAM权重。资产解压到 `/home/tiger/robodojo-data/Assets`，RoboDojo/Assets 指向它；仿真不直接从HDFS逐个读取USD。依赖源码快照也有SHA锁定，与两个主仓库Git版本共同组成运行版本。

基础镜像固定为 `d8h852v1enldjpkjjr7g`；提交时校验baseline返回的镜像VID，不能悄悄更换镜像继续使用旧运行时。

评测基线使用 RoboDojo `b08b49c081953bb3302d079a383c9e059f952f0d`。这与此前r11冻结环境逐文件一致。本地RoboDojo可能有其他开发修改；不使用它们覆盖官方评测环境。EMERGE `robodojo` 分支从已验证评测基线9bd1821建立，保留评测修复，未混入main上的后续场景切换/服务重构。

## 发布与提交

1. 在 `robodojo` 分支修改、验证并提交代码，推到 `git@github.com:loongfeili/EMERGE-Policy.git`。RoboDojo如需改变也先在其fork提交，并显式指定新commit。禁止修改任务判分、布局集合、仿真预算后仍称为同一官方配置。
2. 从干净且已push的工作区发布新的唯一release ID：

```bash
cd /mnt/bn/ic-vlm/personal/loongfei.li03/AgenticVLA08/EMERGE-Policy
python deployment/merlin/publish.py --release-id merlin-v1-YYYYMMDD-NN
```

发布到缓存的 `releases/<release-id>/`。`source-lock.json`固定仓库，`cache-lock.json`固定依赖，`release.json`固定脚本/配置。已存在的release拒绝覆盖。main源代码不从旧tar包恢复。`isaaclab-full.tar.gz`等属于锁定的依赖，不是主仓库替代物。

3. 先查询资源，然后对具体配置dry-run；不带 `--submit` 不创建任务：

```bash
merlin-cli --control-plane cn-seed resource my-group-resources list --json '{"page_size":100,"filter":{"type":"gpu"}}'
python deployment/merlin/launch.py \
  --release /mnt/hdfs/__MERLIN_USER_DIR__/emerge_robodojo_20260925/releases/<release-id> \
  --config verify.json \
  --env-file /home/tiger/.config/emerge-robodojo/merlin-env.json \
  --receipt /tmp/<release-id>-verify-receipt.json
```

确认配置后同一命令加 `--submit`。脚本使用 `job-v2 runs get-request-config`恢复完整job配置，覆盖完整env_map/资源/挂载，调用平台precheck和fork；敏感请求只短暂存于0600临时文件，随后删除。receipt只记录job链接、公开配置和变量名。receipt存在时拒绝重复提交；如停在submitting，先查平台是否已创建，不能因查询超时再提交一份。

新一轮实验按这个顺序执行：发布唯一release → `infer.json` dry-run并提交 → 查询推理job与服务健康 → `verify.json` dry-run并提交 → 核验4条结果 → 按余量选择 `full8.json`、`full16.json`、`full32.json` 或 `full64.json`。每个配置使用独立receipt；同一release的验证与全量任务复用同版本推理服务，输出到不同run目录。下面是推理提交示例，评测只需替换config和receipt：

```bash
python deployment/merlin/launch.py \
  --release /mnt/hdfs/__MERLIN_USER_DIR__/emerge_robodojo_20260925/releases/<new-release-id> \
  --config infer.json \
  --env-file /home/tiger/.config/emerge-robodojo/merlin-env.json \
  --receipt /tmp/<new-release-id>-infer-receipt.json --submit
```

已有封存运行时，不需要为每次实验重新运行 `infer-build.json`。若只重跑已有配置，先确认其run目录和receipt未被使用；已有结果的恢复遵循下文续跑约束。修改配置须在发布源中完成并创建新release，不能直接编辑已封存的JSON或脚本。

模型使用 `gpt-6-astra`，主Agent、定位和验证使用同一模型默认值。API通过私有env文件配置；2026-10-04验证通过的AIDP Responses配置为：

```json
{
  "EMERGE_PROVIDER": "responses",
  "EMERGE_API_BASE": "https://aidp.bytedance.net/api/modelhub/online",
  "EMERGE_MODEL": "gpt-6-astra",
  "EMERGE_REASONING_EFFORT": "high",
  "EMERGE_REASONING_SUMMARY": "auto"
}
```

`AZURE_OPENAI_API_KEY`单独保存在同一0600私有env文件中，公开示例不含key；也兼容已有`EMERGE_API_KEY`（两者同时存在时后者优先）。请求为`POST https://aidp.bytedance.net/api/modelhub/online/responses`，使用Bearer认证、每次请求生成的`X-TT-LOGID`及SSE流式响应。不添加`/v1`或Azure deployment路径，也不使用`api-version`。AIDP走内部网络，不继承资产代理或旧`EMERGE_RESPONSES_PROXY`。主Agent、子Agent及视觉监控均通过同一provider工厂读取这些设置。

2026-10-04实际测试通过：官方SDK的文本、图像、工具调用往返；EMERGE流式文本、看图生成工具参数、工具随机返回值读取及后续多轮历史；部署预检的完整工具往返。发送的reasoning为`{"effort":"high","summary":"auto"}`，服务在SDK响应中回显summary为`detailed`、模型名为`deployment-gpt-6-astra-platform-global`。开发机框架测试中流式文本约3.5秒、图像加工具及后续对话三次请求合计约4.7秒，不能据此推断真实任务或高并发延迟。

预检现在继承实际Agent的推理强度和token预算，验证工具调用后回传随机值并继续回答；不再用`low`和128-token上限替代实际配置。旧Azure Chat入口的工具调用400及错误Responses路径404仍保留为历史诊断；上述正确入口已解决这次阻碍。测试证据在缓存`diagnostics/aidp-responses-20261004/`。

历史release保持不变；未指定`EMERGE_PROVIDER`时仍使用旧`custom`配置及`https://edge.lingsuan.org/v1/responses`，便于复现。旧Responses链路使用`EMERGE_RESPONSES_PROXY`；`EMERGE_ASSET_PROXY`只负责GitHub/NVIDIA资产访问。API接入变更须另发release并做验证，不能据本机连通性测试宣称Merlin节点或全量评测已通过。

用户提供的是SSH公钥，可用于 `VSCODE_SSH_KEY`，不能拿公钥充当Git私钥；当前开发机已有loongfeili的有效SSH认证。Pod读取公开fork使用HTTPS，无需分发私钥。

### 平台环境变量日志限制（实测）

2026-09-28的L20 trial422819582中，平台的 `/opt/tiger/arnold/arnold_entrypoint/entrypoint.sh:29` 在用户脚本执行前无条件运行 `env | LC_ALL=C sort`。因此通过 `env_map` 注入的API key会出现在平台启动stdout；不能将普通env_map称为加密Secret存储。代码、HDFS发布物、应用日志和结果导出不写key，但这不能消除平台更早的环境打印。查询创建schema和平台环境变量文档未找到可验证的关闭/脱敏开关，没有修改或绕过平台启动器。

当前按用户指定的Merlin环境变量接口传参。原始平台日志按敏感资料处理，不复制到公开报告或Git；本次已进入启动日志的key建议由持有人轮换。要彻底消除这一平台日志可见性，需要平台提供日志脱敏或原生Secret注入支持，不能靠在用户入口脚本里unset补救。凭据文档仅在本机0600文件中。

## 推理服务与冷启动验证

推理任务统一名称 `geometry_seg_infer`。默认组1894、cluster44、队列 `a100-sxm-80gb.hpccluster-ydfgrrp7ac9tiffwmqs7.ai`。每4卡为一组：π两副本、VGGT一副本、SAM一副本。`infer.json`为1节点×4卡；`infer8.json`为1节点×8卡；`infer-scale8.json`为2节点×4卡。平台实际设备名称必须记录，不能仅凭队列名断言型号。

首次创建运行时用 `infer-build.json`。此配置允许安装依赖，随后将两个完整venv和包清单封存到该release结果目录的 `runtime-build/`，每个归档SHA回读校验。将这个目录的内容校验后复制到缓存 `runtime/inference-v1/`。普通 `infer.json`仅恢复这个冻结环境，不联网重新解析Python版本。运行时改变应使用新版本路径、新release；不要覆盖已封存目录。

提交推理后，等待 `services/pool.json`：每个节点发布模型地址和源码commit，节点0合并副本池。仿真Pod验证commit一致且每个endpoint实际健康后才继续。不要并行启动两份推理配置写同一release的服务目录；换配置先停止旧推理任务，保留归档。

服务清单通过检查后保存为Pod本地 `/home/tiger/robodojo-setup/services.json`，后续预检和评测读取这个固定副本。共享pool仅在地址集合变化时发布；周期心跳在node清单。这样避免HDFS FUSE动态文件的大小/内容缓存不同步造成JSON解析失败。运行中的地址不热切换；服务故障按失败流程诊断后重启。

`verify.json`采用2节点×1 L20，每卡1 worker，stack_bowls/build_tower各2个布局，共4条，seed0。它验证跨节点分片和共享结果，不用于判断策略总体成功率。每个Pod执行：Git拉取与校验→缓存校验/本地解压→环境变量生成私有配置→真实API工具调用→10步仿真/π推理→π并发请求→VGGT和SAM参考及实景调用→正式episode→视频/轨迹/最终结果写HDFS。

验收必须看实际产物：两节点部署commit一致；api-probe通过；policy action shape正确；perception服务调用通过，几何质量拒绝须独立记录；4条互斥episode、0基础设施错误；每条都有episode_status、session、动作记录和视频；final快照SHA通过；最终coverage完整。任务失败/策略得分0是不同概念，不能将策略失败伪装成基础设施错误，也不能将部分得分算二元成功。

r5实测：两个L20节点均从封存缓存恢复并通过完整54配置/2100布局检查，API工具调用分别4.02秒和3.24秒通过；π输出为50×14，并发4请求约0.47–0.48秒完成。推理服务实际使用4张A800-SXM4-80GB。VGGT/SAM服务请求成功，但参考场景和节点0实景存在多视角定位拒绝，节点1实景存在尺度MAD 0.271超过0.25阈值的拒绝。质量拒绝有明确结果，不绕过阈值；服务部署通过不等于几何定位可用率或算法效果通过。

## 扩容与完整标准评测

|配置|节点×每节点L20|每卡仿真worker|总worker|范围|
|---|---:|---:|---:|---|
|verify.json|2×1|1|2|4个工程验证episode|
|full4w1.json|1×4|1|4|54个任务配置（含变体）、2100原生布局、seed0|
|full8w1.json|1×8|1|8|同上|
|full8.json|1×8|2|16|54个任务配置（含变体）、2100原生布局、seed0|
|full16.json|2×8|2|32|同上|
|full32.json|4×8|2|64|同上|
|full64.json|8×8|2|128|同上|

完整评测 `tasks=all, layouts=native`，任务清单在 `configs/robodojo_tasks_arx_x5_seed0.txt`。按全量episode索引对node数取模分片，不能每节点各跑一次完整集合。默认主Agent40轮，官方每任务仿真上限不变。保持内置skill可读，RoboDojo只开放支持的工具，不调用WAM。修改模型、提示、预算、几何阈值均用新run标识。

API 网关可能先于 GPU 达到并发上限。2026-09-28 的 seed0 r1 在32个仿真worker下出现 `gateway_concurrency_limit`，已停止并保留诊断。`full4w1.json`、`full8w1.json` 仅降低同时运行的episode数，仍覆盖全部2100条；上线前必须实测 API 并发，并检查正式轨迹是否出现基础设施错误。

54个配置由42个基础任务和12个`_random`变体组成，合计2100个seed0布局。官方汇总将变体合并回42个基础任务，再按五个能力维度汇总；54个配置和42个基础任务是不同统计口径。

扩容先保证资源余量，再增加评测节点。观察推理服务请求延迟/队列与GPU利用率；按实际瓶颈增加π或几何副本。当前提供的4卡分组是可复用起点，不是64/128worker吞吐已经通过的声明。增加推理节点后生成新的服务池并做实际并发预检，不仅看healthz。

## 结果、续跑与排障

每次结果在 `<结果根>/<release-id>/runs/<run-id>/`。

首次启动先校验、解压较大的运行时和资产包，再执行预检；出现 `READY_FOR_FULL_EVALUATION` 前没有正式episode结果。正式轨迹在episode完成后同步到HDFS，运行中先看节点状态和 `eval-tail.log`。新文件上传期间或动态文件刷新时可能暂时读到空内容，应等待完成标志和不可变快照，不因一次空读取重启任务。

当前结果写入锁内逐条同步session、图像、定位中间产物和三路视频；同节点的结果上传串行。r5节点1叠碗结束到搭塔批次开始约11分钟，间隔包括同步和仿真进程收尾，未单独拆分耗时；不能将其全部计为上传耗时。同任务的持久worker可以继续运行后续布局，跨任务切换则等待前一批次处理结束。估算全量墙钟时间时需计入同步、任务切换及冷启动，并在增加并发前测量HDFS吞吐与小文件开销。当前工程验证不构成32/64卡存储吞吐已达标的结论。

- `node-NN/deployment.json`：Git版本、配置、服务地址及命令，不含API密钥。
- `node-NN/{api,policy,perception}-probe.json`和包清单：预检证据。
- `node-NN/episodes/.../workspace`及视频目录：按评测器manifest定位每条轨迹，勿猜文件名。
- `node-NN/runner-exit.json`引用不可变 `results-final-<uuid>.jsonl` 及SHA，优先读取此快照。
- `progress.json`是动态视图，HDFS FUSE可能短暂缓存旧值；`final-progress.json`和`final-official-summary.json`用于终态检查。
- `STOP.json`表示共享停止原因；连续基础设施错误、API不可用或共享存储故障会触发停止。

先用 `merlin-cli --control-plane cn-seed job-v2 runs get --json '{"sid":"<job>"}'`查权威状态。停止的job不能继续执行Pod命令。日志沿run→trial→pod查，使用pod SID拉日志。运行中的查询超时不意味着任务结束。

续跑前必须诊断STOP，确认代码、模型、任务列表、seed、Agent预算一致，保留旧STOP和失败证据，再使用同一run身份恢复。评测器只跳过已完成的有效episode；基础设施错误应重试。更改分片数可能让旧node结果重复，不能直接在同一run覆盖；扩缩容应新建run，或另行实现并验证按全局episode集合迁移。不要手动删除失败记录让统计好看。

评测退出后保留HDFS产物，停止本次验证专用推理资源；若后续评测继续使用则明确记录。历史示例433b239fc535fb2b/trial421897160是旧r11流程，最终已失败停止，不能当作当前服务发现源。

## 本地检查

```bash
.venv/bin/python -m pytest -q tests
python -m unittest discover -s deployment/merlin -p 'test*.py' -v
```

前者覆盖评测隔离、API恢复、终止分类、轨迹同步等；后者覆盖固定Git版本、拒绝脏源码、env_map注入/不继承旧密钥、发布物篡改和扩容资源形状。单元测试不替代Merlin冷启动和真实轨迹验收。

终态验收命令（在有ffprobe的开发机执行）：

```bash
python deployment/merlin/verify_run.py \
  --release /mnt/hdfs/__MERLIN_USER_DIR__/emerge_robodojo_20260925/releases/<release-id> \
  --run-dir /mnt/hdfs/_BYTE_DATA_SEED_/ssd_hldy/user/lilongfei.xjgm/emerge_merlin/<release-id>/runs/<run-id> \
  --config verify.json --output /tmp/<run-id>-verification.json
```

推理服务目录的 `node-NN-gpu.json` 每10秒更新实测GPU型号、利用率与显存，服务日志尾部也同步至HDFS。读取时检查 `updated_at`：跨挂载的FUSE缓存可能返回旧内容，时间戳明显滞后时用运行Pod或平台指标核对实时负载。评估吞吐须结合客户端实际RPC延迟和完成速度。API密钥仅注入评测Pod，推理Pod不需要此密钥。
