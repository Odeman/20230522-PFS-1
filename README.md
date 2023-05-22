## Tested Environment
- Ubuntu 16
- C++ 11
- GCC 5.4
- Boost
- cmake

## Compile
```sh
$ cmake .
$ make
```

## Parameters
```sh
./agenda action_name --algo <algorithm> [options]
```
- action:
    - query: static SSPPR query
    - build: build index, for FORA only
    - dynamic-ss: dynamic SSPPR  query
    - dynamic-ss-queue-workload
    - dynamic-ss-queue-workload-parallel
    - dynamic-ss-queue-workload-parallel-shuffling
- algo: 
    - lazyup: Agenda with lazy update
- options



## Data
The example data format is in `./data/webstanford/` folder. The data for DBLP, Pokec, Orkut, LiveJournal, Twitter are not included here for size limitation reason. You can find them online.

## Examples
Generate query files for the graph data. Each line contains a node id.

```sh
$./agenda dynamic-ss --algo lazyup --epsilon 0.5 --prefix ../data/ --dataset Orkut --query_size 100 --update_size 100 --beta 1 --with_idx
$./agenda dynamic-ss-queue-workload --algo lazyup --epsilon 0.5 --prefix ../data/ --dataset LJ --query_arrRate 0.1 --update_arrRate 0.1 --queue_time 1000 --beta 1 --with_idx
$./agenda dynamic-ss-test-vic-num --algo lazyup --epsilon 0.5 --prefix ../data/ --dataset LJ --query_arrRate 0 --update_arrRate 1 --queue_time 30 --beta 1 --degree_shuffling_PFS 5 --with_idx
$./agenda dynamic-ss-queue-workload-parallel --algo lazyup --epsilon 0.5 --prefix ../data/ --dataset LJ --query_arrRate 0.1 --update_arrRate 0.1 --queue_time 1000 --beta 1 --with_idx --total_worker_num 10
$./agenda dynamic-ss-queue-workload-parallel-shuffling --algo lazyup --epsilon 0.5 --prefix ../data/ --dataset LJ --query_arrRate 0.1 --update_arrRate 0.1 --queue_time 1000 --beta 1 --with_idx --total_worker_num 9 --total_shuffling_controller_num 1
```





## Acknowledgement 
Part of the code is reused from FORA's codebase: https://github.com/wangsibovictor/fora
# 20230522-PFS
# 20230522-PFS
