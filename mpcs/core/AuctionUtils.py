"""Versioned DP auction and immutable server serving-quality ledger."""

from __future__ import annotations

from random import Random
from types import MappingProxyType
from typing import Iterable, Mapping

from mpcs.config import AuctionConfig
from mpcs.core.Domain import (
    OpaqueAuctionAward,
    OpaqueAuctionLot,
    SealedBidIntent,
    ServingQualitySnapshot,
    VerifiedSealedIntent,
)
class PaperAuctioneer:
    """Run the configured auction over already-verified opaque intents."""

    __slots__ = ("_config", "_tie_rng")

    def __init__(self, *, config: AuctionConfig, tie_seed: int = 0) -> None:
        config.validate()
        if not isinstance(tie_seed, int):
            raise ValueError("tie_seed must be an integer")
        self._config = config
        self._tie_rng = Random(tie_seed)

    def settle(
        self,
        lots: tuple[OpaqueAuctionLot, ...],
        verified_opaque_intents: tuple[VerifiedSealedIntent, ...],
        serving_quality: ServingQualitySnapshot,
    ) -> tuple[OpaqueAuctionAward, ...]:
        lots_by_token = self._validate_lots(lots)
        copied_intents = tuple(verified_opaque_intents)
        if any(
            type(intent) is not VerifiedSealedIntent
            or type(intent.server_payload) is not SealedBidIntent
            for intent in copied_intents
        ):
            raise TypeError(
                "auction accepts only VerifiedSealedIntent records with "
                "exactly SealedBidIntent payloads"
            )
        if any(
            intent.server_payload.frozen_offer_amount is None
            for intent in copied_intents
        ):
            raise ValueError(
                "reverse-Vickrey auction requires a frozen quality-discounted bid"
            )
        return self._settle_frozen_quality_offers(
            lots_by_token=lots_by_token,
            verified_opaque_intents=copied_intents,
        )

    def _settle_frozen_quality_offers(
        self,
        *,
        lots_by_token: Mapping[str, OpaqueAuctionLot],
        verified_opaque_intents: tuple[VerifiedSealedIntent, ...],
    ) -> tuple[OpaqueAuctionAward, ...]:
        """Reverse-Vickrey ranking over frozen quality-discounted bids.

        Lowest bid wins; the winner is paid the second-lowest bid (or its own
        bid when it is the only candidate).  Exact route feasibility remains a
        private validation concern.
        """

        intents_by_lot = self._validate_and_group_intents(
            verified_opaque_intents,
            lots_by_token=lots_by_token,
            serving_quality=None,
        )
        awards: list[OpaqueAuctionAward] = []
        for parcel_token in sorted(lots_by_token):
            lot = lots_by_token[parcel_token]
            intents = intents_by_lot.get(parcel_token, ())
            if not intents:
                continue
            ranked = sorted(
                intents,
                key=lambda item: (
                    item.server_payload.frozen_offer_amount,
                    item.server_payload.intent_token,
                ),
            )
            winner_intent = self._seeded_tie_winner(ranked)
            winner = winner_intent.server_payload
            winner_bid = winner.frozen_offer_amount
            assert winner_bid is not None
            payment = (
                winner_bid
                if len(ranked) == 1
                else ranked[1].server_payload.frozen_offer_amount
            )
            assert payment is not None
            awards.append(
                OpaqueAuctionAward(
                    parcel_token=parcel_token,
                    winner_intent_token=winner.intent_token,
                    winner_platform_id=winner.bidder_platform_id,
                    payment_amount=payment,
                    winner_bid_amount=winner_bid,
                    valid_bidder_count=len(ranked),
                    decision_frame_id=lot.decision_frame_id,
                )
            )
        return tuple(awards)

    def _seeded_tie_winner(
        self, ranked: list[VerifiedSealedIntent]
    ) -> VerifiedSealedIntent:
        """Select equal top offers using the auction's seeded random stream."""

        top_offer = ranked[0].server_payload.frozen_offer_amount
        tied = tuple(
            item
            for item in ranked
            if item.server_payload.frozen_offer_amount == top_offer
        )
        if len(tied) == 1:
            return tied[0]
        ordered_tied = tuple(
            sorted(tied, key=lambda item: item.server_payload.intent_token)
        )
        rotation = self._tie_rng.randrange(len(ordered_tied))
        return ordered_tied[rotation % len(ordered_tied)]

    @staticmethod
    def _validate_lots(
        lots: tuple[OpaqueAuctionLot, ...],
    ) -> Mapping[str, OpaqueAuctionLot]:
        copied = tuple(lots)
        if any(not isinstance(lot, OpaqueAuctionLot) for lot in copied):
            raise TypeError("auction lots must be OpaqueAuctionLot records")
        lots_by_token = {lot.parcel_token: lot for lot in copied}
        if len(lots_by_token) != len(copied):
            raise ValueError("auction contains a duplicate lot token")
        if len(
            {
                lot.decision_frame_id
                for lot in copied
            }
        ) > 1:
            raise ValueError("auction lots must share one decision frame")
        return MappingProxyType(lots_by_token)

    @staticmethod
    def _validate_and_group_intents(
        intents: tuple[VerifiedSealedIntent, ...],
        *,
        lots_by_token: Mapping[str, OpaqueAuctionLot],
        serving_quality: ServingQualitySnapshot | None,
    ) -> Mapping[str, tuple[VerifiedSealedIntent, ...]]:
        copied = tuple(intents)
        if any(
            not isinstance(intent, VerifiedSealedIntent)
            for intent in copied
        ):
            raise TypeError(
                "auction accepts only VerifiedSealedIntent records"
            )
        intent_tokens: set[str] = set()
        privacy_query_ids: set[str] = set()
        eligibility_tokens: set[str] = set()
        bidder_lot_pairs: set[tuple[str, str]] = set()
        grouped: dict[str, list[VerifiedSealedIntent]] = {}
        for intent in copied:
            payload = intent.server_payload
            if type(payload) is not SealedBidIntent:
                raise TypeError(
                    "verified intent payload must be exactly SealedBidIntent"
                )
            try:
                lot = lots_by_token[payload.parcel_token]
            except KeyError as error:
                raise ValueError(
                    "verified intent references an unknown auction lot"
                ) from error
            if payload.decision_frame_id != lot.decision_frame_id:
                raise ValueError(
                    "verified intent and lot use different decision frames"
                )
            if (
                serving_quality is not None
                and payload.bidder_platform_id
                not in serving_quality.scores_by_platform_id
            ):
                raise ValueError(
                    "serving-quality ledger lacks a bidder score"
                )
            bidder_lot = (
                payload.parcel_token,
                payload.bidder_platform_id,
            )
            if bidder_lot in bidder_lot_pairs:
                raise ValueError(
                    "one platform submitted duplicate bids for a lot"
                )
            if payload.intent_token in intent_tokens:
                raise ValueError("duplicate sealed intent token")
            if payload.privacy_query_id in privacy_query_ids:
                raise ValueError("duplicate privacy query ID")
            if intent.opaque_eligibility_token in eligibility_tokens:
                raise ValueError("duplicate opaque eligibility token")
            bidder_lot_pairs.add(bidder_lot)
            intent_tokens.add(payload.intent_token)
            privacy_query_ids.add(payload.privacy_query_id)
            eligibility_tokens.add(intent.opaque_eligibility_token)
            grouped.setdefault(payload.parcel_token, []).append(intent)
        return MappingProxyType(
            {
                token: tuple(values)
                for token, values in grouped.items()
            }
        )


class StaticServingQualityProvider:
    """One immutable public quality input for the whole experiment.

    The production mechanism deliberately has no reputation update loop:
    quality is a configuration coefficient, not an assignment/completion
    history.  This keeps the public bid input independent of private routing
    and makes every frame observe the same snapshot.
    """

    __slots__ = ("_snapshot",)

    def __init__(
        self,
        *,
        platform_ids: Iterable[str],
        static_service_quality: float,
    ) -> None:
        ordered_platform_ids = tuple(sorted(platform_ids))
        if (
            not ordered_platform_ids
            or len(set(ordered_platform_ids)) != len(ordered_platform_ids)
            or any(
                not isinstance(platform_id, str) or not platform_id
                for platform_id in ordered_platform_ids
            )
        ):
            raise ValueError(
                "serving-quality platform IDs must be unique and non-empty"
            )
        if not 0 <= float(static_service_quality) <= 1:
            raise ValueError("static_service_quality must be in [0, 1]")
        score = float(static_service_quality)
        self._snapshot = ServingQualitySnapshot(
            ledger_version=0,
            scores_by_platform_id={
                platform_id: score
                for platform_id in ordered_platform_ids
            },
            service_reliability_by_platform_id={
                platform_id: score
                for platform_id in ordered_platform_ids
            },
        )

    @classmethod
    def from_config(
        cls,
        *,
        platform_ids: Iterable[str],
        config: AuctionConfig,
    ) -> StaticServingQualityProvider:
        config.validate()
        return cls(
            platform_ids=platform_ids,
            static_service_quality=config.static_service_quality,
        )

    def snapshot(self) -> ServingQualitySnapshot:
        return self._snapshot
