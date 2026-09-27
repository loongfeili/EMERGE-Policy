# Merlin 上的可扩展 RoboDojo / EMERGE 评测

本目录是部署入口。算法代码以 `https://github.com/loongfeili/EMERGE-Policy` 的 `robodojo` 分支为发布源；环境以 `https://github.com/loongfeili/RoboDojo` 为源。每次发布固定完整 commit，每个 Pod 都从 GitHub fetch 并 checkout 该 commit；不把未提交补丁叠在部署源码上。

当前工程流程正在做首次独立验证，验证结果和示例 trial 将在验证完成后写入本文。不要将历史 r11 的“曾经运行”视作新流程已通过。

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

模型固定为 `gpt-6-astra`，服务地址 `https://edge.lingsuan.org`，实际请求 `/v1/responses`。`EMERGE_API_KEY`只从调用者环境或私有env文件读取，主Agent、定位和验证使用同一模型默认值。`EMERGE_RESPONSES_PROXY`用于API代理；`EMERGE_ASSET_PROXY`用于GitHub/NVIDIA资产访问。公开示例不含key。用户提供的是SSH公钥，可用于 `VSCODE_SSH_KEY`，不能拿公钥充当Git私钥；当前开发机已有loongfeili的有效SSH认证。Pod读取公开fork使用HTTPS，无需分发私钥。

## 推理服务与冷启动验证

推理任务统一名称 `geometry_seg_infer`。默认组1894、cluster44、队列 `a100-sxm-80gb.hpccluster-ydfgrrp7ac9tiffwmqs7.ai`。每4卡为一组：π两副本、VGGT一副本、SAM一副本。`infer.json`为1节点×4卡；`infer-scale8.json`为2节点×4卡。平台实际设备名称必须记录，不能仅凭队列名断言型号。

首次创建运行时用 `infer-build.json`。此配置允许安装依赖，随后将两个完整venv和包清单封存到该release结果目录的 `runtime-build/`，每个归档SHA回读校验。将这个目录的内容校验后复制到缓存 `runtime/inference-v1/`。普通 `infer.json`仅恢复这个冻结环境，不联网重新解析Python版本。运行时改变应使用新版本路径、新release；不要覆盖已封存目录。

提交推理后，等待 `services/pool.json`：每个节点发布模型地址和源码commit，节点0合并副本池。仿真Pod验证commit一致且每个endpoint实际健康后才继续。不要并行启动两份推理配置写同一release的服务目录；换配置先停止旧推理任务，保留归档。

`verify.json`采用2节点×1 L20，每卡1 worker，stack_bowls/build_tower各2个布局，共4条，seed0。它验证跨节点分片和共享结果，不用于判断策略总体成功率。每个Pod执行：Git拉取与校验→缓存校验/本地解压→环境变量生成私有配置→真实API工具调用→10步仿真/π推理→π并发请求→VGGT和SAM参考及实景调用→正式episode→视频/轨迹/最终结果写HDFS。

验收必须看实际产物：两节点部署commit一致；api-probe通过；policy action shape正确；perception服务调用通过，几何质量拒绝须独立记录；4条互斥episode、0基础设施错误；每条都有episode_status、session、动作记录和视频；final快照SHA通过；最终coverage完整。任务失败/策略得分0是不同概念，不能将策略失败伪装成基础设施错误，也不能将部分得分算二元成功。

## 扩容与完整标准评测

|配置|节点×每节点L20|每卡仿真worker|总worker|范围|
|---|---:|---:|---:|---|
|verify.json|2×1|1|2|4个工程验证episode|
|full8.json|1×8|2|16|54个任务配置（含变体）、2100原生布局、seed0|
|full32.json|4×8|2|64|同上|
|full64.json|8×8|2|128|同上|

完整评测 `tasks=all, layouts=native`，任务清单在 `configs/robodojo_tasks_arx_x5_seed0.txt`。按全量episode索引对node数取模分片，不能每节点各跑一次完整集合。默认主Agent40轮，官方每任务仿真上限不变。保持内置skill可读，RoboDojo只开放支持的工具，不调用WAM。修改模型、提示、预算、几何阈值均用新run标识。

扩容先保证资源余量，再增加评测节点。观察推理服务请求延迟/队列与GPU利用率；按实际瓶颈增加π或几何副本。当前提供的4卡分组是可复用起点，不是64/128worker吞吐已经通过的声明。增加推理节点后生成新的服务池并做实际并发预检，不仅看healthz。

## 结果、续跑与排障

每次结果在 `<结果根>/<release-id>/runs/<run-id>/`：

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

推理服务目录的 `node-NN-gpu.json` 每10秒更新实测GPU型号、利用率与显存，服务日志尾部也同步至HDFS。它是当前负载快照；评估吞吐须结合客户端实际RPC延迟和完成速度。API密钥仅注入评测Pod，推理Pod不需要此密钥。
