"""Pluggable non-learning baseline components for the common environment.

The classes in this module deliberately stop at the domain protocols.  They
choose actions, local route insertions, and opaque cross-platform offers; the
environment remains responsible for validation, settlement, movement, and
completion accounting.
"""

from __future__ import annotations

import random
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping, Sequence

if TYPE_CHECKING:
    from .Framework import _LocalAlgorithm


from mpcs.core.Domain import (
    CandidateIntentBundle,
    CandidatePrivateReceipt,
    CrossBidder,
    OpaqueAuctionAward,
    OpaqueAuctionLot,
    PlatformPlanningSnapshot,
    PublicParcelDescriptor,
    SealedBidIntent,
    ServingQualitySnapshot,
    VehicleSnapshot,
    VehicleStatus,
    VerifiedSealedIntent,
)
from mpcs.core.GraphUtils import RegionIndex


from .Common import (
    _opaque_id,
    _stable_seed,
)


class RegionalPriorityCrossBidder(CrossBidder):
    """Privacy-preserving bidder backed by region-scoped EV handles.

    ``pool-random`` and ``residual-random`` publish one opaque platform bundle
    whose private receipt contains the region's EV handles.  The environment
    samples one handle from the global pool before private feasibility
    validation.
    The legacy ``all``/``first-random`` modes remain available for the
    auction-backed paths.
    """

    __slots__ = (
        "platform_id",
        "_region_index",
        "_sharing_rate",
        "_seed",
        "_candidate_mode",
        "_query_index",
        "_token_registry",
        "_local_algorithm",
    )

    def __init__(
        self,
        *,
        platform_id: str,
        region_index: RegionIndex,
        sharing_rate: float = 0.3,
        seed: int = 0,
        candidate_mode: str = "all",
        token_registry: Any | None = None,
        local_algorithm: _LocalAlgorithm | None = None,
    ) -> None:
        self.platform_id = platform_id
        self._region_index = region_index
        self._sharing_rate = float(sharing_rate)
        self._seed = int(seed)
        self._candidate_mode = candidate_mode
        self._query_index = 0
        self._token_registry = token_registry
        self._local_algorithm = local_algorithm
        if not 0.0 <= self._sharing_rate <= 1.0:
            raise ValueError("sharing_rate must be in [0, 1]")
        if candidate_mode not in {
            "all",
            "first-random",
            "pool-random",
            "residual-random",
        }:
            raise ValueError("unknown candidate mode")

    @property
    def candidate_mode(self) -> str:
        """Private construction mode used by the environment seam."""
        return self._candidate_mode

    @property
    def cross_selection_mode(self) -> str:
        """Whether the environment must make one global random draw."""
        if self._candidate_mode in {"pool-random", "residual-random"}:
            return "random-single"
        return "auction"

    def build_intents(
        self,
        public_snapshot: Any,
        own_planning_state: PlatformPlanningSnapshot,
    ) -> tuple[CandidateIntentBundle, ...]:
        if own_planning_state.platform_id != self.platform_id:
            raise ValueError("bidder received foreign planning state")
        locally_reserved = (
            self._local_algorithm.local_vehicle_ids_for_frame(own_planning_state.frame)
            if self._local_algorithm is not None
            else frozenset()
        )
        if self._candidate_mode == "pool-random":
            # Sampling must see every current same-Region EV handle.  Status,
            # capacity, route, and cost are private validation facts and must
            # not pre-filter the uniform draw.
            available = tuple(
                sorted(
                    own_planning_state.vehicles,
                    key=lambda item: item.vehicle_id,
                )
            )
        else:
            available = tuple(
                vehicle
                for vehicle in sorted(
                    own_planning_state.vehicles,
                    key=lambda item: item.vehicle_id,
                )
                if (
                    vehicle.vehicle_id not in locally_reserved
                    and vehicle.status is not VehicleStatus.RETURNING
                    and vehicle.load_count < vehicle.max_capacity
                )
            )
        by_region: dict[str, tuple[str, ...]] = {}
        vehicles_by_region: dict[str, list[VehicleSnapshot]] = {}
        for vehicle in available:
            region_id = self._vehicle_region(vehicle)
            if region_id is not None:
                vehicles_by_region.setdefault(region_id, []).append(vehicle)
        by_region = {
            region_id: tuple(vehicle.vehicle_id for vehicle in vehicles)
            for region_id, vehicles in vehicles_by_region.items()
        }
        bundles: list[CandidateIntentBundle] = []
        for descriptor in sorted(
            public_snapshot.descriptors, key=lambda item: item.parcel_token
        ):
            if self._token_registry is not None and self._token_registry.owns(
                descriptor.parcel_token
            ):
                continue
            candidate_ids = by_region.get(descriptor.obfuscated_region_id, ())
            candidates = vehicles_by_region.get(descriptor.obfuscated_region_id, ())
            if not candidates:
                continue
            if self._candidate_mode == "first-random":
                rng = random.Random(
                    _stable_seed(self._seed, self.platform_id, descriptor.parcel_token)
                )
                rng.shuffle(candidates)
                candidates = candidates[:1]
            payment = float(descriptor.fare_amount) * self._sharing_rate
            if payment <= 0.0:
                continue
            candidate_groups = (
                (candidate_ids,)
                if self._candidate_mode in {"pool-random", "residual-random"}
                else (tuple(candidates),)
            )
            for candidate_group in candidate_groups:
                intent_token = _opaque_id(
                    "baseline-intent",
                    self.platform_id,
                    descriptor.parcel_token,
                    str(self._query_index),
                )
                self._query_index += 1
                if self._candidate_mode in {"pool-random", "residual-random"}:
                    offer = payment
                else:
                    count = len(by_region.get(descriptor.obfuscated_region_id, ()))
                    # Lower offers win in the shared reverse auction.  This
                    # legacy encoding is never used by random fixed-payment
                    # strategies.
                    epsilon = payment / (1000.0 * (count + 1))
                    offer = max(payment - epsilon * count, payment * 0.5)
                bundles.append(
                    CandidateIntentBundle(
                        server_payload=SealedBidIntent(
                            intent_token=intent_token,
                            parcel_token=descriptor.parcel_token,
                            bidder_platform_id=self.platform_id,
                            privacy_query_id=f"baseline-query:{intent_token}",
                            decision_frame_id=descriptor.decision_frame_id,
                            frozen_offer_amount=offer,
                        ),
                        private_receipt=CandidatePrivateReceipt(
                            intent_token=intent_token,
                            bidder_platform_id=self.platform_id,
                            true_potential_component=0.0,
                            true_beta=0.0,
                            private_candidate_vehicle_ids=(
                                candidate_group
                                if self._candidate_mode
                                in {"pool-random", "residual-random"}
                                else tuple(
                                    vehicle.vehicle_id for vehicle in candidate_group
                                )
                            ),
                        ),
                    )
                )
        return tuple(bundles)

    def _vehicle_region(self, vehicle: VehicleSnapshot) -> str | None:
        region_id = self._region_index.region_id_for_point(vehicle.current_location)
        if region_id is not None:
            return region_id
        return self._region_index.region_id_for_node_or_none(
            vehicle.current_road_node_id
        )


class FixedPaymentRegionalAuctioneer:
    """Select the lowest encoded regional offer and pay sharing_rate * fare."""

    __slots__ = ("sharing_rate", "seed")

    def __init__(self, *, sharing_rate: float = 0.3, seed: int = 0) -> None:
        self.sharing_rate = float(sharing_rate)
        self.seed = int(seed)

    def settle(
        self,
        lots: tuple[OpaqueAuctionLot, ...],
        verified_opaque_intents: tuple[VerifiedSealedIntent, ...],
        serving_quality: ServingQualitySnapshot,
    ) -> tuple[OpaqueAuctionAward, ...]:
        del serving_quality
        lots_by_token = {lot.parcel_token: lot for lot in lots}
        grouped: dict[str, list[VerifiedSealedIntent]] = {}
        for intent in verified_opaque_intents:
            grouped.setdefault(intent.server_payload.parcel_token, []).append(intent)
        awards: list[OpaqueAuctionAward] = []
        for token in sorted(lots_by_token):
            candidates = sorted(
                grouped.get(token, ()),
                key=lambda item: (
                    item.server_payload.frozen_offer_amount,
                    item.server_payload.intent_token,
                ),
            )
            if not candidates:
                continue
            min_offer = candidates[0].server_payload.frozen_offer_amount
            tied = [
                item
                for item in candidates
                if item.server_payload.frozen_offer_amount == min_offer
            ]
            rng = random.Random(_stable_seed(self.seed, token))
            winner = tied[rng.randrange(len(tied))].server_payload
            payment = lots_by_token[token].fare_amount * self.sharing_rate
            if payment <= 0.0 or winner.frozen_offer_amount is None:
                continue
            if winner.frozen_offer_amount > payment:
                continue
            awards.append(
                OpaqueAuctionAward(
                    parcel_token=token,
                    winner_intent_token=winner.intent_token,
                    winner_platform_id=winner.bidder_platform_id,
                    payment_amount=payment,
                    winner_bid_amount=winner.frozen_offer_amount,
                    valid_bidder_count=len(candidates),
                    decision_frame_id=lots_by_token[token].decision_frame_id,
                )
            )
        return tuple(awards)


class SeededRandomFixedPaymentAuctioneer(FixedPaymentRegionalAuctioneer):
    """Fed-LTD fixed-payment adapter after the global EV draw."""

    def settle(
        self,
        lots: tuple[OpaqueAuctionLot, ...],
        verified_opaque_intents: tuple[VerifiedSealedIntent, ...],
        serving_quality: ServingQualitySnapshot,
    ) -> tuple[OpaqueAuctionAward, ...]:
        # The environment has already made the one global EV draw.  Reuse
        # the fixed-payment contract without another candidate selection.
        return super().settle(lots, verified_opaque_intents, serving_quality)


def _build_default_release_sanitizers(
    *,
    platform_ids: Sequence[str],
    region_index: RegionIndex | None,
    seed: int,
) -> Mapping[str, Any]:
    """Build a private, deterministic region-only sanitizer for ablations."""
    if region_index is None:
        raise ValueError("prepared must provide region_index for release sanitization")
    global_region_ids = frozenset(region.region_id for region in region_index.regions)
    sanitizers: dict[str, Any] = {}
    for platform_id in platform_ids:
        sanitizers[platform_id] = _BaselineRegionReleaseSanitizer(
            platform_id=platform_id,
            global_region_ids=global_region_ids,
            seed=seed,
        )
    return MappingProxyType(sanitizers)


def _build_default_serving_quality_provider(
    *,
    platform_ids: Sequence[str],
    config: Any,
) -> Any:
    """Use configured static quality when available, otherwise neutral quality."""
    auction = getattr(config, "auction", None)
    score = float(getattr(auction, "static_service_quality", 0.5))
    return _BaselineServingQualityProvider(
        platform_ids=platform_ids,
        static_service_quality=score,
    )


class _BaselineTokenRegistry:
    """Small private token registry used by the default non-FL sanitizer."""

    __slots__ = ("platform_id", "_tokens")

    def __init__(self, platform_id: str) -> None:
        self.platform_id = platform_id
        self._tokens: set[str] = set()

    def register(self, token: str) -> None:
        if token in self._tokens:
            raise ValueError("duplicate baseline parcel token")
        self._tokens.add(token)

    def owns(self, token: str) -> bool:
        return token in self._tokens


class _BaselineRegionReleaseSanitizer:
    """Region-only release adapter with no dependency on the FL privacy stack."""

    __slots__ = (
        "platform_id",
        "_global_region_ids",
        "_seed",
        "_query_counts",
        "_token_registry",
    )

    def __init__(
        self,
        *,
        platform_id: str,
        global_region_ids: frozenset[str],
        seed: int,
    ) -> None:
        self.platform_id = platform_id
        self._global_region_ids = global_region_ids
        self._seed = seed
        self._query_counts: dict[tuple[str, str], int] = {}
        self._token_registry = _BaselineTokenRegistry(platform_id)

    @property
    def token_registry(self) -> _BaselineTokenRegistry:
        return self._token_registry

    def sanitize(self, own_release_view: Any) -> tuple[PublicParcelDescriptor, ...]:
        if own_release_view.platform_id != self.platform_id:
            raise ValueError("baseline sanitizer received foreign platform")
        descriptors: list[PublicParcelDescriptor] = []
        for parcel in sorted(own_release_view.parcels, key=lambda item: item.parcel_id):
            if parcel.region_id not in self._global_region_ids:
                raise ValueError("baseline sanitizer received unknown region")
            key = (own_release_view.frame.episode_id, parcel.parcel_id)
            query_index = self._query_counts.get(key, 0)
            self._query_counts[key] = query_index + 1
            token = _opaque_id(
                "baseline-release",
                str(self._seed),
                self.platform_id,
                own_release_view.frame.episode_id,
                own_release_view.frame.decision_frame_id,
                parcel.parcel_id,
                str(query_index),
            )
            self._token_registry.register(token)
            descriptors.append(
                PublicParcelDescriptor(
                    parcel_token=token,
                    obfuscated_region_id=parcel.region_id,
                    fare_amount=parcel.fare_amount,
                    decision_frame_id=own_release_view.frame.decision_frame_id,
                )
            )
        return tuple(descriptors)


class _BaselineServingQualityProvider:
    """Immutable quality snapshot used when no configured provider is passed."""

    __slots__ = ("_snapshot",)

    def __init__(
        self, *, platform_ids: Sequence[str], static_service_quality: float
    ) -> None:
        self._snapshot = ServingQualitySnapshot(
            ledger_version=0,
            scores_by_platform_id={
                platform_id: static_service_quality for platform_id in platform_ids
            },
            service_reliability_by_platform_id={
                platform_id: static_service_quality for platform_id in platform_ids
            },
        )

    def snapshot(self) -> ServingQualitySnapshot:
        return self._snapshot
