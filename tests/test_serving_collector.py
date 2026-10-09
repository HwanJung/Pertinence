import json

from pertinence.serving.collector import filter_cadvisor, pod_inventory


def test_collector_keeps_only_required_namespace_counters() -> None:
    payload = "\n".join(
        [
            '# HELP container_cpu_usage_seconds_total test',
            'container_cpu_usage_seconds_total{namespace="pertinence-serving",pod="a"} 1',
            'container_memory_working_set_bytes{namespace="pertinence-serving",pod="a"} 2',
            'container_cpu_cfs_throttled_seconds_total{namespace="other",pod="b"} 3',
            'unrelated_metric{namespace="pertinence-serving",pod="a"} 4',
        ]
    )
    assert filter_cadvisor(payload) == [
        'container_cpu_usage_seconds_total{namespace="pertinence-serving",pod="a"} 1',
        'container_memory_working_set_bytes{namespace="pertinence-serving",pod="a"} 2',
    ]


def test_pod_inventory_keeps_uid_and_total_restart_count() -> None:
    payload = json.dumps(
        {
            "items": [
                {
                    "metadata": {"name": "expert-resnet56-predictor-a", "uid": "uid-a"},
                    "status": {
                        "containerStatuses": [
                            {"name": "kserve-container", "restartCount": 1},
                            {"name": "queue-proxy", "restartCount": 2},
                        ]
                    },
                }
            ]
        }
    ).encode()

    assert pod_inventory(payload) == [
        {"name": "expert-resnet56-predictor-a", "uid": "uid-a", "restart_count": 3}
    ]
