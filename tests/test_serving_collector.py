from pertinence.serving.collector import filter_cadvisor


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
