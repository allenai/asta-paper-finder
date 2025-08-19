#!/bin/bash

make_foreach_parallel() {
    local count=$#
    local subcommand="${!count}"             # Last argument is the subcommand
    local dirs=("${@:1:count-1}")            # All other arguments are directories

    local pids=()
    local dir_names=()
    local results=()
    
    # Launch all processes in background, keep track of dirs and pids
    for dir in "${dirs[@]}"; do
        (
            cd "$dir" && make "$subcommand" &> /dev/null
        ) &
        pids+=($!)
        dir_names+=("$dir")
    done

    # Wait and collect results
    for i in "${!pids[@]}"; do
        wait "${pids[$i]}"
        results[$i]=$?
    done

    local fail=0
    local failed_dirs=()
    for i in "${!results[@]}"; do
        if [[ ${results[$i]} -ne 0 ]]; then
            ((fail++))
            failed_dirs+=("${dir_names[$i]}")
        fi
    done

    if [[ $fail -eq 0 ]]; then
        echo "✅ All Successful"
    else
        echo "$fail failed:"
        for dir in "${failed_dirs[@]}"; do
            echo "   ❌ $dir"
        done
    fi
}

make_foreach_sequential() {
    local count=$#
    local subcommand="${!count}"             # Last argument is the subcommand
    local dirs=("${@:1:count-1}")            # All other arguments are directories

    for dir in "${dirs[@]}"; do
        (
            cd "$dir" && make "$subcommand"
        )
        local res=$?
        if [[ $res -ne 0 ]]; then
            echo "❌ Failed in $dir (exit code $res)"
            return $res
        fi
    done

    echo "✅ All Successful"
    return 0
}


make_foreach() {
    local mode="$1"
    shift

    # Get the subcommand: the last argument after shifting off 'mode'
    local subcommand="${!#}"
    case "$mode" in
        par)
            echo "** Running in PARALLEL mode (subcommand: $subcommand) **"
            make_foreach_parallel "$@"
            ;;
        seq)
            echo "** Running in SEQUENTIAL mode (subcommand: $subcommand) **"
            make_foreach_sequential "$@"
            ;;
        *)
            echo "MODE must be either 'par' or 'seq'"
            return 1
            ;;
    esac
}
