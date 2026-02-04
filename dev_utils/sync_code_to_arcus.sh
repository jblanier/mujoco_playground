rsync -vr \
--exclude .pixi \
--exclude pixi.lock \
--exclude results \
--exclude outputs \
--exclude tests \
--exclude wandb \
--exclude logs \
--exclude learning\logs \
--exclude .venv \
--exclude *.mp4 \
--exclude *.lock \
~/git/mujoco_playground arcus-18:git/

rsync -vr \
--exclude .pixi \
--exclude pixi.lock \
--exclude results \
--exclude outputs \
--exclude tests \
--exclude wandb \
--exclude logs \
--exclude learning\logs \
--exclude .venv \
--exclude *.mp4 \
--exclude *.lock \
~/git/brax arcus-18:git/
