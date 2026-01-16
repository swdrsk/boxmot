# Mikel Broström 🔥 BoxMOT 🧾 AGPL-3.0 license

"""
Minimal BaseTrack implementation for SimplePreSort.
SimplePreSort doesn't use complex track state management.
"""


class TrackState:
    """Track state enumeration (simplified for SimplePreSort)"""
    New = 0
    Tracked = 1


class BaseTrack:
    """
    Minimal base track class for SimplePreSort.
    
    SimplePreSort doesn't maintain track states across frames,
    so this is a minimal implementation.
    """
    _count = 0
    track_id = 0
    is_activated = False
    state = TrackState.New

    @staticmethod
    def next_id():
        """Get next track ID"""
        BaseTrack._count += 1
        return BaseTrack._count

    @staticmethod
    def clear_count():
        """Reset track ID counter"""
        BaseTrack._count = 0

    def activate(self, *args):
        """Activate track (placeholder)"""
        self.track_id = self.next_id()
        self.is_activated = True
        self.state = TrackState.Tracked

    def mark_removed(self):
        """Mark track as removed (placeholder)"""
        self.is_activated = False
