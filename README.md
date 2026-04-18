# Product Inspection System

UT 产品点云检测系统 V1 首版工程骨架。

## Layout

- `client/`: Windows 桌面客户端
- `server/`: FastAPI API 网关与文件化任务状态管理
- `workers/`: 本地线程/Celery 任务编排与 Pointcept、pc-skeletor 适配器
- `shared/`: 公共数据模型、配置、产品型号注册表、校验逻辑
- `tests/`: 单元测试

## Quick Start

```powershell
cd product-inspection-system
python -m uvicorn server.app:app --host 0.0.0.0 --port 8000
```

客户端：

```powershell
cd product-inspection-system
python -m client.app
```

## Environment Variables

- `PIS_RUNTIME_ROOT`: 运行期目录，默认 `product-inspection-system/.runtime`
- `PIS_SERVER_URL`: 客户端 API 地址，默认 `http://127.0.0.1:8000`
- `PIS_API_TOKEN`: 可选 API Token
- `PIS_MAX_UPLOAD_MB`: 上传大小限制，默认 `512`
- `PIS_USE_CELERY`: 为 `1` 时尝试使用 Celery 派发任务
- `PIS_CELERY_BROKER_URL`: Celery broker
- `PIS_CELERY_RESULT_BACKEND`: Celery backend
- `PIS_COLMAP_COMMAND`: 图像重建命令模板，支持 `{image_dir}` `{output_dir}`
- `PIS_3DGS_COMMAND`: 3DGS 命令模板，支持 `{image_dir}` `{output_dir}`

## Existing Assets

首版默认指向：

- `Pointcept/exp/byme/semseg-pt-v3m1-0-base-36parts/config.py`
- `Pointcept/exp/byme/semseg-pt-v3m1-0-base-36parts/model/model_last.pth`
- `pc-skeletor/tools/skeletonize_pointcept_instances.py`
- `pc-skeletor/tools/compute_skeleton_curve_lengths.py`

如果你的模型权重路径不同，修改 `shared/product_models/byme_36.json` 即可。
