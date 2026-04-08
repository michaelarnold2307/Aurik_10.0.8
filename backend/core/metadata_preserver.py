"""backend/core/metadata_preserver.py — Audio metadata preservation across export.

Copies ID3/Vorbis/RIFF tags (artist, album, title, year, genre, cover art,
track number) from source file to exported file.  Adds Aurik restoration
provenance tag (SHA-256 of original + Aurik version + timestamp).

Singleton access: ``get_metadata_preserver()``

Supported tag families (via mutagen):
  - ID3     (MP3)
  - Vorbis  (FLAC, OGG)
  - RIFF    (WAV — INFO chunks; limited)
  - AIFF    (ID3 via mutagen)

Dependencies: mutagen (pure-Python, no C ext).
"""

from __future__ import annotations

import hashlib
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_instance: MetadataPreserver | None = None
_lock = threading.Lock()


def get_metadata_preserver() -> MetadataPreserver:
    """Thread-safe singleton access (double-checked locking)."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = MetadataPreserver()
    return _instance


try:
    import mutagen
    from mutagen.aiff import AIFF
    from mutagen.flac import FLAC, Picture
    from mutagen.id3 import APIC, ID3, TALB, TCON, TDRC, TIT2, TPE1, TRCK
    from mutagen.mp3 import MP3
    from mutagen.oggvorbis import OggVorbis

    _MUTAGEN_AVAILABLE = True
except ImportError:
    _MUTAGEN_AVAILABLE = False
    logger.warning("mutagen not installed — metadata preservation disabled")


# Canonical tag mapping: internal key → (ID3 frame, Vorbis key)
_TAG_MAP: dict[str, tuple[str, str]] = {
    "title": ("TIT2", "TITLE"),
    "artist": ("TPE1", "ARTIST"),
    "album": ("TALB", "ALBUM"),
    "date": ("TDRC", "DATE"),
    "genre": ("TCON", "GENRE"),
    "tracknumber": ("TRCK", "TRACKNUMBER"),
}


@dataclass
class AudioMetadata:
    """Extracted audio metadata for transfer between files."""

    title: str = ""
    artist: str = ""
    album: str = ""
    date: str = ""
    genre: str = ""
    tracknumber: str = ""
    cover_art: bytes | None = None
    cover_mime: str = "image/jpeg"
    extra: dict[str, str] = field(default_factory=dict)

    def has_content(self) -> bool:
        """Return True if any meaningful tag is set."""
        return bool(self.title or self.artist or self.album or self.date or self.genre)


class MetadataPreserver:
    """Extracts metadata from source audio and applies it to exported files."""

    def extract(self, source_path: str | Path) -> AudioMetadata:
        """Extract metadata from source audio file.

        Parameters
        ----------
        source_path : str | Path
            Path to the original audio file.

        Returns
        -------
        AudioMetadata
            Extracted tags (empty fields if unreadable or mutagen unavailable).
        """
        meta = AudioMetadata()
        if not _MUTAGEN_AVAILABLE:
            return meta

        src = Path(source_path)
        if not src.exists():
            return meta

        try:
            mf = mutagen.File(str(src))
            if mf is None:
                return meta
        except Exception as exc:
            logger.debug("metadata extract failed for %s: %s", src.name, exc)
            return meta

        # --- ID3-based (MP3, AIFF) ---
        if isinstance(mf, (MP3, AIFF)) or (hasattr(mf, "tags") and isinstance(getattr(mf, "tags", None), ID3)):
            tags = mf.tags
            if tags is None:
                return meta
            meta.title = str(tags.get("TIT2", "") or "")
            meta.artist = str(tags.get("TPE1", "") or "")
            meta.album = str(tags.get("TALB", "") or "")
            meta.date = str(tags.get("TDRC", "") or "")
            meta.genre = str(tags.get("TCON", "") or "")
            meta.tracknumber = str(tags.get("TRCK", "") or "")
            # Cover art (first APIC frame)
            for key in tags:
                if key.startswith("APIC"):
                    apic = tags[key]
                    if hasattr(apic, "data"):
                        meta.cover_art = apic.data
                        meta.cover_mime = getattr(apic, "mime", "image/jpeg")
                    break

        # --- Vorbis-based (FLAC, OGG) ---
        elif isinstance(mf, (FLAC, OggVorbis)):
            for internal_key, (_id3_frame, vorbis_key) in _TAG_MAP.items():
                vals = mf.get(vorbis_key)
                if vals:
                    setattr(meta, internal_key, str(vals[0]))
            # Cover art from FLAC pictures
            if isinstance(mf, FLAC) and mf.pictures:
                pic = mf.pictures[0]
                meta.cover_art = pic.data
                meta.cover_mime = pic.mime

        # --- Generic fallback ---
        elif hasattr(mf, "tags") and mf.tags:
            tags = mf.tags
            for internal_key, (_id3_frame, vorbis_key) in _TAG_MAP.items():
                # Try vorbis-style first, then ID3
                val = tags.get(vorbis_key) or tags.get(_id3_frame)
                if val:
                    if isinstance(val, list):
                        val = val[0]
                    setattr(meta, internal_key, str(val))

        logger.debug(
            "metadata extracted: artist=%s, title=%s, album=%s",
            meta.artist[:30] if meta.artist else "-",
            meta.title[:30] if meta.title else "-",
            meta.album[:30] if meta.album else "-",
        )
        return meta

    def apply(
        self,
        target_path: str | Path,
        metadata: AudioMetadata,
        *,
        aurik_version: str = "",
        original_hash: str = "",
    ) -> bool:
        """Apply metadata tags to an exported audio file.

        Parameters
        ----------
        target_path : str | Path
            Path to the exported (already written) audio file.
        metadata : AudioMetadata
            Tags to write.
        aurik_version : str
            Aurik version string for provenance tag.
        original_hash : str
            SHA-256 hex digest of original audio for provenance.

        Returns
        -------
        bool
            True if tags were successfully written.
        """
        if not _MUTAGEN_AVAILABLE:
            return False

        tgt = Path(target_path)
        if not tgt.exists():
            return False

        ext = tgt.suffix.lower()
        try:
            if ext == ".mp3":
                return self._apply_id3(tgt, metadata, aurik_version, original_hash)
            elif ext == ".flac":
                return self._apply_flac(tgt, metadata, aurik_version, original_hash)
            elif ext in (".ogg", ".oga"):
                return self._apply_vorbis(tgt, metadata, aurik_version, original_hash)
            elif ext in (".aiff", ".aif"):
                return self._apply_aiff(tgt, metadata, aurik_version, original_hash)
            else:
                logger.debug("metadata apply: unsupported format %s", ext)
                return False
        except Exception as exc:
            logger.warning("metadata apply failed for %s: %s", tgt.name, exc)
            return False

    def transfer(
        self,
        source_path: str | Path,
        target_path: str | Path,
        *,
        aurik_version: str = "",
    ) -> bool:
        """Extract metadata from source and apply to target in one call.

        Also computes SHA-256 provenance hash of the source file.
        """
        meta = self.extract(source_path)
        if not meta.has_content() and meta.cover_art is None:
            logger.debug("metadata transfer: no tags found in %s", Path(source_path).name)
            # Still write provenance even without original tags
            if aurik_version:
                orig_hash = self._file_hash(source_path)
                return self.apply(target_path, AudioMetadata(), aurik_version=aurik_version, original_hash=orig_hash)
            return False

        orig_hash = self._file_hash(source_path) if aurik_version else ""
        return self.apply(target_path, meta, aurik_version=aurik_version, original_hash=orig_hash)

    # ── Private: format-specific writers ──────────────────────────────────

    def _provenance_comment(self, aurik_version: str, original_hash: str) -> str:
        """Build provenance string for embedding in tags."""
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        parts = [f"Restored by Aurik {aurik_version}" if aurik_version else "Restored by Aurik"]
        parts.append(f"Date: {ts}")
        if original_hash:
            parts.append(f"Original-SHA256: {original_hash[:16]}…")
        return " | ".join(parts)

    def _apply_id3(self, path: Path, meta: AudioMetadata, version: str, orig_hash: str) -> bool:
        try:
            tags = ID3(str(path))
        except mutagen.id3.ID3NoHeaderError:
            tags = ID3()

        if meta.title:
            tags.add(TIT2(encoding=3, text=[meta.title]))
        if meta.artist:
            tags.add(TPE1(encoding=3, text=[meta.artist]))
        if meta.album:
            tags.add(TALB(encoding=3, text=[meta.album]))
        if meta.date:
            tags.add(TDRC(encoding=3, text=[meta.date]))
        if meta.genre:
            tags.add(TCON(encoding=3, text=[meta.genre]))
        if meta.tracknumber:
            tags.add(TRCK(encoding=3, text=[meta.tracknumber]))
        if meta.cover_art:
            tags.add(APIC(encoding=3, mime=meta.cover_mime, type=3, desc="Cover", data=meta.cover_art))
        if version:
            from mutagen.id3 import COMM

            tags.add(
                COMM(
                    encoding=3, lang="eng", desc="Aurik Provenance", text=[self._provenance_comment(version, orig_hash)]
                )
            )

        tags.save(str(path))
        logger.info("metadata applied (ID3): %s", path.name)
        return True

    def _apply_flac(self, path: Path, meta: AudioMetadata, version: str, orig_hash: str) -> bool:
        mf = FLAC(str(path))
        for internal_key, (_id3_frame, vorbis_key) in _TAG_MAP.items():
            val = getattr(meta, internal_key, "")
            if val:
                mf[vorbis_key] = [val]
        if version:
            mf["COMMENT"] = [self._provenance_comment(version, orig_hash)]
        if meta.cover_art:
            pic = Picture()
            pic.data = meta.cover_art
            pic.mime = meta.cover_mime
            pic.type = 3  # Cover (front)
            mf.add_picture(pic)
        mf.save()
        logger.info("metadata applied (FLAC): %s", path.name)
        return True

    def _apply_vorbis(self, path: Path, meta: AudioMetadata, version: str, orig_hash: str) -> bool:
        mf = OggVorbis(str(path))
        for internal_key, (_id3_frame, vorbis_key) in _TAG_MAP.items():
            val = getattr(meta, internal_key, "")
            if val:
                mf[vorbis_key] = [val]
        if version:
            mf["COMMENT"] = [self._provenance_comment(version, orig_hash)]
        mf.save()
        logger.info("metadata applied (Vorbis): %s", path.name)
        return True

    def _apply_aiff(self, path: Path, meta: AudioMetadata, version: str, orig_hash: str) -> bool:
        mf = AIFF(str(path))
        if mf.tags is None:
            mf.add_tags()
        tags = mf.tags
        if meta.title:
            tags.add(TIT2(encoding=3, text=[meta.title]))
        if meta.artist:
            tags.add(TPE1(encoding=3, text=[meta.artist]))
        if meta.album:
            tags.add(TALB(encoding=3, text=[meta.album]))
        if meta.date:
            tags.add(TDRC(encoding=3, text=[meta.date]))
        if meta.genre:
            tags.add(TCON(encoding=3, text=[meta.genre]))
        if meta.tracknumber:
            tags.add(TRCK(encoding=3, text=[meta.tracknumber]))
        if meta.cover_art:
            tags.add(APIC(encoding=3, mime=meta.cover_mime, type=3, desc="Cover", data=meta.cover_art))
        if version:
            from mutagen.id3 import COMM

            tags.add(
                COMM(
                    encoding=3, lang="eng", desc="Aurik Provenance", text=[self._provenance_comment(version, orig_hash)]
                )
            )
        mf.save()
        logger.info("metadata applied (AIFF/ID3): %s", path.name)
        return True

    @staticmethod
    def _file_hash(path: str | Path, chunk_size: int = 65536) -> str:
        """Compute SHA-256 of a file (first 1 MB for speed)."""
        h = hashlib.sha256()
        remaining = 1024 * 1024  # 1 MB cap for speed
        try:
            with open(path, "rb") as f:
                while remaining > 0:
                    chunk = f.read(min(chunk_size, remaining))
                    if not chunk:
                        break
                    h.update(chunk)
                    remaining -= len(chunk)
        except OSError:
            return ""
        return h.hexdigest()
