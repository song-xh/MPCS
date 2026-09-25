"""Pluggable non-learning baseline components for the common environment.

The classes in this module deliberately stop at the domain protocols.  They
choose actions, local route insertions, and opaque cross-platform offers; the
environment remains responsible for validation, settlement, movement, and
completion accounting.
"""

from __future__ import annotations

from typing import Sequence


from mpcs.core.Domain import (
    LocalAssignmentProposal,
    PlatformPlanningSnapshot,
    PickupPlanningRequest,
    RoutePlanningService,
)


from .Common import (
    _option_priority,
    _proposal,
    _shadow_after,
)


class LocalSumRule:
    def _localsum(
        self,
        requests: Sequence[PickupPlanningRequest],
        state: PlatformPlanningSnapshot,
        planning: RoutePlanningService,
    ) -> tuple[LocalAssignmentProposal, ...]:
        shadow = state
        proposals: list[LocalAssignmentProposal] = []
        for request in requests:
            option = min(
                self._available_options(request, shadow, planning),
                key=_option_priority,
                default=None,
            )
            if option is None:
                continue
            proposals.append(
                _proposal(
                    frame=state.frame,
                    platform_id=self.platform_id,
                    request=request,
                    option=option,
                    method=self.method,
                )
            )
            shadow = _shadow_after(shadow, option)
        return tuple(proposals)
