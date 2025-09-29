import asyncio, os, shutil
from datetime import date
from dataclasses import dataclass

from app.infrastructure.settings import Settings
from app.infrastructure.pg.dao import (
    pg_expired_campaign_ids,
    pg_delete_videos_by_campaign,
    pg_delete_campaign_retention,
)
from app.infrastructure.keyframes.cache_fs import _dir_for as _kf_dir

@dataclass
class RetentionCleaner:
    settings: Settings
    _stop: bool = False

    async def run_forever(self):
        """Loop periódico que elimina videos de campañas cuyo end_date ya pasó."""
        interval = max(5, int(self.settings.CLEANUP_INTERVAL_MIN)) * 60
        while not self._stop:
            try:
                await self._run_once()
            except Exception as e:
                # Aquí podrías loggear el error
                pass
            await asyncio.sleep(interval)

    async def _run_once(self):
        today = date.today()
        expired = pg_expired_campaign_ids(today)
        if not expired:
            return

        base_dir = self.settings.KEYFRAMES_DIR
        for cid in expired:
            # 1) borra de Postgres y obtén video_ids
            video_ids = pg_delete_videos_by_campaign(cid)

            # 2) limpia keyframes en FS
            for vid in video_ids:
                try:
                    d = _kf_dir(vid, base_dir)
                    shutil.rmtree(d, ignore_errors=True)
                except Exception:
                    pass

            # 3) borra la fila de retención
            try:
                pg_delete_campaign_retention(cid)
            except Exception:
                pass

    def stop(self):
        self._stop = True