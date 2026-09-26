"""A decision-only algorithm that can join comparisons or mixed training."""

from mpcs.core.Domain import ParcelAction


def local_first(_config, _platform_id, observation):
    return {
        pickup.parcel_id: ParcelAction.LOCAL
        for pickup in observation.waiting_pickups
    }


def register(runner):
    runner.register_policy("local-first", local_first)
