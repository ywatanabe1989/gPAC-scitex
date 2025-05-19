#!/bin/bash
# Time-stamp: "2024-11-05 21:42:41 (ywatanabe)"
# File: ./torchPAC/scripts/run_experiment.sh


LOG_FILE="$0.log"
start=$(date +"%Y%m%d_%H%M%S")

usage() {
    echo "Usage: $0 [-h|--help]"
    echo "Options:"
    echo " -h, --help Display this help message"
    echo
    echo "Example:"
    echo " $0"
    exit 1
}

run_experiment() {
    # Executes the experiment workflow
    #
    # Usage:
    # run_experiment

    _initialize

    # Start logging CPU / GPU usages
    ./scripts/record_processor_usages.py --interval_s 0.33 --init &

    # PAC calculation with stats recording
    ./scripts/main.py

    end=$(date +"%Y%m%d_%H%M%S")

    # Close
    _kill_py

    # 
    cp -v /tmp/processor_usages.csv ./data/processor_usages-"$start"-"$end".csv
    # # Plot the metrics
    # ./scripts/plot_parallel.py &&


}

_initialize() {
    sudo nvidia-smi -pm 1 >/dev/null 2>&1
    # Preparation
    _kill_py
    rm -f /tmp/mngs/processor_usages.csv
    rm -f /tmp/processor_usages.csv    
    # rm -rf \
    #    ./scripts/main/FINISHED* \
    #    ./scripts/main/RUNNING/ \
    #    /tmp/mngs/processor_usages.csv 
}

_kill_py() {
    # Terminates all Python processes for the current user
    ps aux | grep "$(whoami)" | grep 'python' | awk '{ print "kill", $2 }' | sh
}

main() {
    while [[ "$#" -gt 0 ]]; do
        case $1 in
            -h|--help) usage ;;
            *) echo "Unknown parameter passed: $1"; usage ;;
        esac
        shift
    done

    run_experiment
}

main "$@" 2>&1 | tee "$LOG_FILE"

# EOF
