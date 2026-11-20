"""
CRD Writer
----------
Converts QPUTelemetry objects into QuantumProcessor CRD manifests
and applies them to Kubernetes via kubectl or the k8s Python client.

When K8S_APPLY=false (default), prints the manifest as YAML for inspection.
When K8S_APPLY=true, applies to the cluster using the in-cluster or
kubeconfig credentials.
"""

import json
import logging
import subprocess
import tempfile
import os
from datetime import datetime, timezone
from typing import Optional

import yaml

from backends.simulator_backend import QPUTelemetry, telemetry_age_minutes
from config import K8S_NAMESPACE, K8S_APPLY, STALENESS_THRESHOLD_MINUTES

logger = logging.getLogger(__name__)


def telemetry_to_crd(telemetry: QPUTelemetry,
                     quality_forecast: Optional[list] = None) -> dict:
    """
    Convert a QPUTelemetry object to a QuantumProcessor CRD dict.

    Args:
        quality_forecast: optional list of QualityWindow objects (or dicts)
            produced by the Predictive Quality Forecaster; written to
            status.qualityForecast.
    """
    now_dt = datetime.now(timezone.utc)
    now = now_dt.isoformat()

    # CRD name: sanitize backend name (k8s names must be lowercase alphanumeric + dash)
    crd_name = telemetry.backend_name.lower().replace("_", "-")

    conditions = []

    # TelemetryFresh condition: probe succeeded AND the snapshot is recent
    age = telemetry_age_minutes(telemetry, now_dt)
    if not telemetry.available:
        fresh, reason = False, "ProbeFailed"
    elif age > STALENESS_THRESHOLD_MINUTES:
        fresh, reason = False, "TelemetryStale"
    else:
        fresh, reason = True, "ProbeSucceeded"
    conditions.append({
        "type": "TelemetryFresh",
        "status": "True" if fresh else "False",
        "lastTransitionTime": now,
        "reason": reason,
    })

    # BelowFidelityThreshold condition (warn if fidelity < 0.99)
    below_threshold = telemetry.mean_qubit_fidelity < 0.99 and not telemetry.is_simulator_fallback
    conditions.append({
        "type": "BelowFidelityThreshold",
        "status": "True" if below_threshold else "False",
        "lastTransitionTime": now,
        "reason": "LowFidelity" if below_threshold else "FidelityOK",
    })

    # HighQueueDepth condition (warn if queue > 30)
    high_queue = telemetry.queue_depth > 30
    conditions.append({
        "type": "HighQueueDepth",
        "status": "True" if high_queue else "False",
        "lastTransitionTime": now,
        "reason": "QueueCongested" if high_queue else "QueueOK",
    })

    manifest = {
        "apiVersion": "quantum.openshift.io/v1alpha1",
        "kind": "QuantumProcessor",
        "metadata": {
            "name": crd_name,
            "namespace": K8S_NAMESPACE,
            "labels": {
                "quantum.openshift.io/provider": telemetry.provider,
                "quantum.openshift.io/backend": telemetry.backend_name,
                "quantum.openshift.io/is-simulator": str(telemetry.is_simulator).lower(),
                "quantum.openshift.io/is-fallback": str(telemetry.is_simulator_fallback).lower(),
            },
            "annotations": {
                "quantum.openshift.io/last-probe": now,
            }
        },
        "spec": {
            "provider": telemetry.provider,
            "backendName": telemetry.backend_name,
            "qubitCount": telemetry.qubit_count,
            "topology": telemetry.topology,
            "isSimulator": telemetry.is_simulator,
            "pollingIntervalSeconds": 30,
        },
        "status": {
            "lastUpdated": telemetry.last_updated,
            "calibrationTimestamp": telemetry.calibration_timestamp,
            "calibrationAgeMinutes": telemetry.calibration_age_minutes,
            "available": telemetry.available,
            "physicalProperties": {
                "meanQubitFidelity":          telemetry.mean_qubit_fidelity,
                "minQubitFidelity":           telemetry.min_qubit_fidelity,
                "maxQubitFidelity":           telemetry.max_qubit_fidelity,
                "meanT1Microseconds":         telemetry.mean_t1_us,
                "minT1Microseconds":          telemetry.min_t1_us,
                "meanT2Microseconds":         telemetry.mean_t2_us,
                "minT2Microseconds":          telemetry.min_t2_us,
                "meanSingleQubitGateErrorRate": telemetry.mean_sq_error,
                "maxSingleQubitGateErrorRate":  telemetry.max_sq_error,
                "meanTwoQubitGateErrorRate":    telemetry.mean_tq_error,
                "maxTwoQubitGateErrorRate":     telemetry.max_tq_error,
                "meanReadoutErrorRate":        telemetry.mean_ro_error,
                "queueDepth":                 telemetry.queue_depth,
                "estimatedQueueWaitMinutes":   telemetry.estimated_queue_wait_minutes,
                "costPerShot":                 telemetry.cost_per_shot,
                "nativeGateSet":               telemetry.native_gates,
                "couplingMap":                 telemetry.coupling_map or [],
                "operationDurations":          telemetry.durations or {},
            },
            "pricing": telemetry.pricing or {},
            "qualityForecast": [
                w.to_dict() if hasattr(w, "to_dict") else dict(w)
                for w in (quality_forecast or [])
            ],
            "conditions": conditions,
        }
    }
    return manifest


def apply_crd(telemetry: QPUTelemetry, quality_forecast: Optional[list] = None) -> bool:
    """
    Convert telemetry to CRD manifest and apply it.
    Returns True on success.
    """
    manifest = telemetry_to_crd(telemetry, quality_forecast)
    yaml_str = yaml.dump(manifest, default_flow_style=False, sort_keys=False)

    if not K8S_APPLY:
        # Dry-run: just print
        print("\n" + "="*60)
        print(f"[DRY-RUN] QuantumProcessor CRD for: {telemetry.backend_name}")
        print("="*60)
        print(yaml_str)
        return True

    # Apply to Kubernetes via kubectl
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml',
                                         delete=False) as f:
            f.write(yaml_str)
            tmp_path = f.name

        result = subprocess.run(
            ["kubectl", "apply", "-f", tmp_path],
            capture_output=True, text=True, timeout=30
        )
        os.unlink(tmp_path)

        if result.returncode == 0:
            logger.info(f"[CRDWriter] Applied CRD for {telemetry.backend_name}: "
                        f"{result.stdout.strip()}")
            return True
        else:
            logger.error(f"[CRDWriter] kubectl apply failed for {telemetry.backend_name}: "
                         f"{result.stderr.strip()}")
            return False

    except FileNotFoundError:
        logger.error("[CRDWriter] kubectl not found. Set K8S_APPLY=false for dry-run mode.")
        return False
    except Exception as e:
        logger.error(f"[CRDWriter] Unexpected error applying CRD: {e}")
        return False
