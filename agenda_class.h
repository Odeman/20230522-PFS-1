#ifndef AGENDA_CLASS
#define AGENDA_CLASS
#define _CRT_SECURE_NO_DEPRECATE
#include "mylib.h"
#include <chrono>
#include <iostream>
#include <fstream>
#include <map>
#include <stdlib.h>
#include <set>
#include <string>
#include <time.h>
#include <stdlib.h>
#include <vector>
#include <algorithm>
#include "graph.h"
#include "config.h"
#include "algo.h"
#include "query.h"
#include "build.h"
#include <sys/stat.h>

#include <boost/progress.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <chrono>


using namespace std::chrono;

using namespace boost;
using namespace boost::property_tree;

using namespace std;

class Agenda_class{

public:
    protected:
    Fwdidx fwd_idx;
    Bwdidx bwd_idx;
    ReversePushidx reverse_idx;
private:
    vector<int> q;  //nodes that can still propagate forward
    int *q_array; //nodes that can still propagate forward
    vector<double> fwd_idx_reserve;
    vector<double> fwd_idx_residue;
    NewGraph &graph;
    double epsilon;
    double omega;
    //vector<double> &inacc_idx;
    int this_worker_number;
    vector<int> ppr;
    std::minstd_rand rd;

    double check_total_rsum_all_queries;
    double total_queries;
    double this_worker_omega; //related to num_random walk
    double this_worker_epsilon; //related to num_random walk

//----------------voids------------------------------
public:
    Agenda_class(NewGraph &_graph, double _epsilon, int _this_worker_number): graph(_graph), epsilon(_epsilon), this_worker_number(_this_worker_number)
    {
        init();
    }

    void init(){
        int graph_n=graph.getNumOfVertices();
        q.reserve(graph_n+10);
        q.assign(graph_n, 0);

        fwd_idx.first.initialize(graph_n);
        fwd_idx.second.initialize(graph_n);

        fwd_idx_reserve.reserve(graph_n);
        fwd_idx_residue.reserve(graph_n);
        fwd_idx_reserve.assign(graph_n, 0);
        fwd_idx_residue.assign(graph_n, 0);

        ppr.reserve(graph_n);
        ppr.assign(graph_n, 0);
        omega=0;
        rd.seed(this_worker_number*this_worker_number+10);
        check_total_rsum_all_queries=0;
        total_queries=0;
        q_array=NULL;
    }

//-------------------------------------------------------------------------------------------------------------
    void forward_local_update_linear_class(int v, double &rsum, double rmax, double init_residual = 1.0){
        fwd_idx_reserve.assign(graph.getNumOfVertices(), 0);
        fwd_idx_residue.assign(graph.getNumOfVertices(), 0);
        static vector<bool> idx(graph.getNumOfVertices());
        std::fill(idx.begin(), idx.end(), false);

        double myeps = rmax;//config.rmax;

        vector<int> q;  //nodes that can still propagate forward
        int graph_n=graph.getNumOfVertices();
        q.reserve(graph_n);
        unsigned long left = 0;
        unsigned long left_round=0;
        unsigned long right=0;
        unsigned long right_round=0;
        q[left]=v;

        // residual[s] = init_residual;
        fwd_idx_residue[v]= init_residual;
        
        idx[v] = true;
        
        while (left+graph_n*left_round<=right+right_round*graph_n) {
            int v = q[left];
            idx[v] = false;
            left++;
            if(left>=graph_n){
                left=0;
                left_round++;
            }
            double v_residue = fwd_idx_residue[v];
            fwd_idx_residue[v] = 0;
            fwd_idx_reserve[v] += v_residue * config.alpha;

            rsum -=v_residue*config.alpha;
            const VertexIdType &idx_start = graph.get_neighbor_list_start_pos(v);
            const VertexIdType &idx_end = graph.get_neighbor_list_start_pos(v + 1);
            const VertexIdType degree = idx_end - idx_start;
            const double increment = (1 - config.alpha) * v_residue / degree;
            for (int j = idx_start; j < idx_end; ++j) {
                const VertexIdType &next = graph.getOutNeighbor(j);
                fwd_idx_residue[next] += increment;
                
                if (idx[next] != true) {  
                    const VertexIdType &idx_start_next = graph.get_neighbor_list_start_pos(next);
                    const VertexIdType &idx_end_next = graph.get_neighbor_list_start_pos(next + 1);
                    const VertexIdType degree_next = idx_end_next - idx_start_next;
                    if(fwd_idx_residue[next]/degree_next >= myeps){
                        idx[next] = true;//(int) q.size();
                        right++;
                        if(right>=graph_n){
                            right=0;
                            right_round++;
                        }
                        q[right]=next;
                        //q.push_back(next);   
                    } 
                }
            }
        }
    }
    //------------------------------------Error Push----------------------------------------------------------
    void forward_local_update_linear_error_push_class(int v, double &rsum, double rmax, int update_number, int error_index, double &error_Shuffling, double inaccuracy_bound, double error_Shuffling_bound, double init_residual = 1.0){
        fwd_idx_reserve.assign(graph.getNumOfVertices(), 0);
        fwd_idx_residue.assign(graph.getNumOfVertices(), 0);
        static vector<bool> idx(graph.getNumOfVertices());
        std::fill(idx.begin(), idx.end(), false);

        double myeps = rmax;//config.rmax;

        vector<int> q;  //nodes that can still propagate forward
        int graph_n=graph.getNumOfVertices();
        q.reserve(graph_n);
        unsigned long left = 0;
        unsigned long left_round=0;
        unsigned long right=0;
        unsigned long right_round=0;
        q[left]=v;

        // residual[s] = init_residual;
        fwd_idx_residue[v]= init_residual;
        
        idx[v] = true;
        error_Shuffling=1-inacc_idx_map_worker.inacc[error_index][v];
        vector<int> out_neighbor_temp;
        out_neighbor_temp.reserve(graph.getNumOfVertices());
        int out_neighbor_size=0;

        while (left+graph_n*left_round<=right+right_round*graph_n) {
            int v = q[left];
            idx[v] = false;
            left++;
            if(left>=graph_n){
                left=0;
                left_round++;
            }
            double v_residue = fwd_idx_residue[v];
            fwd_idx_residue[v] = 0;
            fwd_idx_reserve[v] += v_residue * config.alpha;

            rsum -=v_residue*config.alpha;
            error_Shuffling-=v_residue*(1-inacc_idx_map_worker.inacc[error_index][v]);
            if(Workers_batch_recorder.node_type[v]==0){
                const VertexIdType &idx_start = graph.get_neighbor_list_start_pos(v);
                const VertexIdType &idx_end = graph.get_neighbor_list_start_pos(v + 1);
                const VertexIdType degree = idx_end - idx_start;
                
                
                out_neighbor_size=0;
                const double increment = (1 - config.alpha) * v_residue / degree;
                for (int j = idx_start; j < idx_end; ++j) {
                    const VertexIdType &next = graph.getOutNeighbor(j);
                    fwd_idx_residue[next] += increment;
                    error_Shuffling+=increment*(1-inacc_idx_map_worker.inacc[error_index][next]);
                    if (idx[next] != true) {  
                        const VertexIdType &idx_start_next = graph.get_neighbor_list_start_pos(next);
                        const VertexIdType &idx_end_next = graph.get_neighbor_list_start_pos(next + 1);
                        const VertexIdType degree_next = idx_end_next - idx_start_next;
                        if(fwd_idx_residue[next]/degree_next >= myeps){//
                            idx[next] = true;//(int) q.size();
                            right++;
                            if(right>=graph_n){
                                right=0;
                                right_round++;
                            }
                            q[right]=next;
                            //q.push_back(next);   
                        } 
                    }
                }
            }
            else if (Workers_batch_recorder.node_type[v]==1){
                const VertexIdType &idx_start = graph.get_neighbor_list_start_pos(v);
                const VertexIdType &idx_end = graph.get_neighbor_list_start_pos(v + 1);
                const VertexIdType degree = idx_end - idx_start;
                out_neighbor_size=0;
        
                for (int j = idx_start; j < idx_end; ++j) {
                    const VertexIdType &next = graph.getOutNeighbor(j);
                    out_neighbor_temp[out_neighbor_size]=next;
                    out_neighbor_size++;
                }
                bool insert;
                int temp_u, temp_v;
                for(int update_temp=0; update_temp<update_number; update_temp++){
                    insert=true;
                    temp_u=Workers_batch_recorder.update_start[update_temp];
                    if(temp_u==v){
                        temp_v=Workers_batch_recorder.update_end[update_temp];
                        for(int neighbor_temp=0; neighbor_temp<out_neighbor_size; neighbor_temp++){
                            if(out_neighbor_temp[neighbor_temp]==temp_v){
                                insert=false;
                                //out_neighbor_temp.erase(out_neighbor_temp.begin()+neighbor_temp);
                                for(int k=neighbor_temp; k<out_neighbor_size-1; k++){
                                    out_neighbor_temp[k]=out_neighbor_temp[k+1];
                                }
                                out_neighbor_size--;
                                //out_neighbor_size--;
                                break;
                            }
                        }
                        if(insert=true){
                            out_neighbor_temp[out_neighbor_size]=temp_v;
                            out_neighbor_size++;
                            //out_neighbor++;
                        }
                    }
                }
                const double increment = (1 - config.alpha) * v_residue /out_neighbor_size;
                for (int j = 0; j < out_neighbor_size; ++j) {
                    VertexIdType next = out_neighbor_temp[j];
                    fwd_idx_residue[next] += increment;
                    error_Shuffling+=increment*(1-inacc_idx_map_worker.inacc[error_index][next]);
                    if (idx[next] != true) {  
                        const VertexIdType &idx_start_next = graph.get_neighbor_list_start_pos(next);
                        const VertexIdType &idx_end_next = graph.get_neighbor_list_start_pos(next + 1);
                        const VertexIdType degree_next = idx_end_next - idx_start_next;
                        if(fwd_idx_residue[next]/degree_next >= myeps){
                            idx[next] = true;//(int) q.size();
                            right++;
                            if(right>=graph_n){
                                right=0;
                                right_round++;
                            }
                            q[right]=next;
                            //q.push_back(next);   
                        } 
                    }
                }

            }
        }
        left = 0;
        left_round=0;
        right=0;
        right_round=0;
        double check_victim_error=0;
        double check_total_error=0;
        int check_rmax_issue=0;
        for(int i=0; i<graph.getNumOfVertices(); i++){
            const VertexIdType &idx_start_next = graph.get_neighbor_list_start_pos(i);
            const VertexIdType &idx_end_next = graph.get_neighbor_list_start_pos(i + 1);
            const VertexIdType degree_next = idx_end_next - idx_start_next;
            if((1-inacc_idx_map_worker.inacc[error_index][i])>(inaccuracy_bound/degree_next)){
                q[right]=i;
                right++;
                check_victim_error+=fwd_idx_residue[i]*(1-inacc_idx_map_worker.inacc[error_index][i]);
            }
            if(fwd_idx_residue[i]>myeps){
                check_rmax_issue++;
            }
            check_total_error+=fwd_idx_residue[i]*(1-inacc_idx_map_worker.inacc[error_index][i]);
        }
        double error_push_start=omp_get_wtime();
        // printf("check error_Shuffling: %.12f\n", error_Shuffling);
        // printf("check error_victim: %.12f\n", check_victim_error);
        // printf("check error_total: %.12f\n", check_total_error);
        // printf("check ramx_issue: %d\n", check_rmax_issue);
        // printf("check error_Shuffling_bound: %.12f\n", error_Shuffling_bound);
        // printf("check victim number: %d\n", right);
        int push_round=0;
        while(error_Shuffling>error_Shuffling_bound){
            push_round++;
            if(this_worker_number==1&&push_round<=20&&push_round>=0){
                //printf("ErrPush: %d, ErrShuffling: %.12f\n", push_round, error_Shuffling);
            }
            if(push_round>=100){
                break;
            }
            for(int left=0; left<right; left++){
                if(error_Shuffling<=error_Shuffling_bound){
                    break;
                }
                
                int v = q[left];
                
                double v_residue = fwd_idx_residue[v];
                if(v_residue==0){
                    continue;
                }
                fwd_idx_residue[v] = 0;
                fwd_idx_reserve[v] += v_residue * config.alpha;

                rsum -=v_residue*config.alpha;
                error_Shuffling-=v_residue*(1-inacc_idx_map_worker.inacc[error_index][v]);
                if(Workers_batch_recorder.node_type[v]==0){
                    const VertexIdType &idx_start = graph.get_neighbor_list_start_pos(v);
                    const VertexIdType &idx_end = graph.get_neighbor_list_start_pos(v + 1);
                    const VertexIdType degree = idx_end - idx_start;
                    
                    
                    out_neighbor_size=0;
                    const double increment = (1 - config.alpha) * v_residue / degree;
                    for (int j = idx_start; j < idx_end; ++j) {
                        const VertexIdType &next = graph.getOutNeighbor(j);
                        fwd_idx_residue[next] += increment;
                        error_Shuffling+=increment*(1-inacc_idx_map_worker.inacc[error_index][next]);
                        if (idx[next] != true) {  
                            const VertexIdType &idx_start_next = graph.get_neighbor_list_start_pos(next);
                            const VertexIdType &idx_end_next = graph.get_neighbor_list_start_pos(next + 1);
                            const VertexIdType degree_next = idx_end_next - idx_start_next;
                            
                        }
                    }
                }
                else if (Workers_batch_recorder.node_type[v]==1){
                    const VertexIdType &idx_start = graph.get_neighbor_list_start_pos(v);
                    const VertexIdType &idx_end = graph.get_neighbor_list_start_pos(v + 1);
                    const VertexIdType degree = idx_end - idx_start;
                    out_neighbor_size=0;
            
                    for (int j = idx_start; j < idx_end; ++j) {
                        const VertexIdType &next = graph.getOutNeighbor(j);
                        out_neighbor_temp[out_neighbor_size]=next;
                        out_neighbor_size++;
                    }
                    bool insert;
                    int temp_u, temp_v;
                    for(int update_temp=0; update_temp<update_number; update_temp++){
                        insert=true;
                        temp_u=Workers_batch_recorder.update_start[update_temp];
                        if(temp_u==v){
                            temp_v=Workers_batch_recorder.update_end[update_temp];
                            for(int neighbor_temp=0; neighbor_temp<out_neighbor_size; neighbor_temp++){
                                if(out_neighbor_temp[neighbor_temp]==temp_v){
                                    insert=false;
                                    //out_neighbor_temp.erase(out_neighbor_temp.begin()+neighbor_temp);
                                    for(int k=neighbor_temp; k<out_neighbor_size-1; k++){
                                        out_neighbor_temp[k]=out_neighbor_temp[k+1];
                                    }
                                    out_neighbor_size--;
                                    //out_neighbor_size--;
                                    break;
                                }
                            }
                            if(insert=true){
                                out_neighbor_temp[out_neighbor_size]=temp_v;
                                out_neighbor_size++;
                                //out_neighbor++;
                            }
                        }
                    }
                    const double increment = (1 - config.alpha) * v_residue /out_neighbor_size;
                    for (int j = 0; j < out_neighbor_size; ++j) {
                        VertexIdType next = out_neighbor_temp[j];
                        fwd_idx_residue[next] += increment;
                        error_Shuffling+=increment*(1-inacc_idx_map_worker.inacc[error_index][next]);
                        if (idx[next] != true) {  
                            const VertexIdType &idx_start_next = graph.get_neighbor_list_start_pos(next);
                            const VertexIdType &idx_end_next = graph.get_neighbor_list_start_pos(next + 1);
                            const VertexIdType degree_next = idx_end_next - idx_start_next;
                            
                        }
                    }

                }
            }
        }
        double error_push_end=omp_get_wtime();
        printf("check error_push_time: %.12f\n", error_push_end-error_push_start);
    }
    //----------------------------------------------------------------------------------------------------
    static bool err_cmp_class(const pair<int,double> a,const pair<int,double> b){
        return a.second > b.second;
    }
    inline void update_idx_class(int source){
        unsigned long num_rw = rw_idx_info[source].second;
        //if(config.with_baton == true)
            //num_rw = ceil(graph.g[source].size()*config.beta/config.alpha);
        unsigned long begin_idx = rw_idx_info[source].first;
        unsigned long k;
        unsigned long destination;
        int cur;
        for(unsigned long i=0; i<num_rw; i++){ //for each node, do some rand-walks
            const VertexIdType &idx_start = graph.get_in_neighbor_list_start_pos(source);
            const VertexIdType &idx_end = graph.get_in_neighbor_list_start_pos(source + 1);
            const VertexIdType degree = idx_end - idx_start;
            if(degree==0){
                rw_idx_all[this_worker_number][begin_idx+i]=source;
                continue;
            }
            cur = source;
            while (true) {
                if (((double)(rd()%100)/(double)(100))<config.alpha) {
                    destination = cur;
                    break;
                }
                const VertexIdType &idx_start_cur = graph.get_in_neighbor_list_start_pos(source);
                const VertexIdType &idx_end_cur = graph.get_in_neighbor_list_start_pos(source + 1);
                const VertexIdType degree_cur = idx_end - idx_start;
                if (degree_cur){
                    k = (rd())%degree_cur;
                    cur = graph.getOutNeighbor(idx_start_cur+k);
                }
                else{
                    cur = source;
                }
            };
            // rw_idx[source].push_back(destination);
            rw_idx_all[this_worker_number][begin_idx+i]=destination;
        }
        
    }
    void lazy_update_fwdidx_class(double theta){
        if(config.no_rebuild)
            return;
        vector< pair<int,double> > error_idx;
        double rsum=0;
        double errsum=0;
        double inaccsum=0;
        int graph_n=graph.getNumOfVertices();
        // for(long i=0; i<fwd_idx.first.occur.m_num; i++){
        //     int node_id = fwd_idx.first.occur[i];
        for(long i=0; i<graph_n; i++){
            int node_id = i;
            double reserve = fwd_idx_reserve[node_id ];
            double residue = fwd_idx_residue[ node_id ];
            if(residue*(1-inacc_idx_all[this_worker_number][node_id])>0){
                error_idx.emplace_back(make_pair(node_id,residue*(1-inacc_idx_all[this_worker_number][node_id])));
                rsum+=residue;
                errsum+=residue*(1-inacc_idx_all[this_worker_number][node_id]);
            }
        }
        sort(error_idx.begin(), error_idx.end(), err_cmp_class);
        long i=0;
        //double errbound=config.epsilon/graph.n/2/rsum*(1-theta);
        double errbound=config.epsilon/graph.getNumOfVertices()*(1-theta);
        INFO(errsum);
        INFO(errbound);
        while(errsum>errbound){
            //cout<<i<<" : "<<errsum<<"--"<<error_idx[i].second<<endl;;
            update_idx_class(error_idx[i].first);
            inacc_idx_all[this_worker_number][error_idx[i].first]=1;
            errsum-=error_idx[i].second;
            i++;
        }
        //INFO(i);
    }

    void lazy_update_fwdidx_class_shuffling(double theta){
        if(config.no_rebuild)
            return;
        vector< pair<int,double> > error_idx;
        double rsum=0;
        double errsum=0;
        double inaccsum=0;
        int graph_n=graph.getNumOfVertices();
        // for(long i=0; i<fwd_idx.first.occur.m_num; i++){
        //     int node_id = fwd_idx.first.occur[i];
        for(long i=0; i<graph_n; i++){
            int node_id = i;
            double reserve = fwd_idx_reserve[node_id ];
            double residue = fwd_idx_residue[ node_id ];
            if(residue*(1-inacc_idx_all[this_worker_number][node_id])>0){
                error_idx.emplace_back(make_pair(node_id,residue*(1-inacc_idx_all[this_worker_number][node_id])));
                rsum+=residue;
                errsum+=residue*(1-inacc_idx_all[this_worker_number][node_id]);
            }
        }
        sort(error_idx.begin(), error_idx.end(), err_cmp_class);
        long i=0;
        //double errbound=config.epsilon/graph.n/2/rsum*(1-theta);
        double errbound=config.epsilon/graph.getNumOfVertices()*(1-theta)*(1-config.beta_PFS);
        INFO(errsum);
        INFO(errbound);
        while(errsum>errbound){
            //cout<<i<<" : "<<errsum<<"--"<<error_idx[i].second<<endl;;
            update_idx_class(error_idx[i].first);
            inacc_idx_all[this_worker_number][error_idx[i].first]=1;
            errsum-=error_idx[i].second;
            i++;
        }
        //INFO(i);
    }

    int random_walk_class(int source){
        const VertexIdType &idx_start = graph.get_in_neighbor_list_start_pos(source);
        const VertexIdType &idx_end = graph.get_in_neighbor_list_start_pos(source + 1);
        const VertexIdType degree = idx_end - idx_start;
        if(degree==0){
            return source;
        }
        VertexIdType cur = source;
        while (true) {
            if (((double)(rd()%100)/(double)(100))<config.alpha) {
                return cur;
            }
            const VertexIdType &idx_start_cur = graph.get_in_neighbor_list_start_pos(source);
            const VertexIdType &idx_end_cur = graph.get_in_neighbor_list_start_pos(source + 1);
            const VertexIdType degree_cur = idx_end - idx_start;
            if (degree_cur){
                int k = (rd())%degree_cur;
                cur = graph.getOutNeighbor(idx_start_cur+k);
            }
            else{
                cur = source;
            }
        };
    }

    void compute_ppr_with_fwdidx_class(double check_rsum){
        int graph_n=graph.getNumOfVertices();
        ppr.assign(graph_n, 0);
        int node_id;
        double reserve;
        for(long i=0; i< graph_n; i++){
            node_id = i;
            reserve = fwd_idx_reserve[ node_id ];
            ppr[node_id] = reserve;
        }
        //INFO(ppr.occur.m_num);

        // INFO("rsum is:", check_rsum);
        if(check_rsum <= 0.0)
            return;

        unsigned long long num_random_walk = this_worker_omega*check_rsum;
        printf("check rsum: %.12f\n", check_rsum);
        printf("check num of allll RW: %d\n", num_random_walk);
        //INFO(num_random_walk);
        //num_total_rw += num_random_walk;
        int extra_random_walk=0;
        double total_residual=0;
        int total_RW=0;
        {
            //Timer timer(RONDOM_WALK); //both rand-walk and source distribution are related with num_random_walk
            //Timer tm(SOURCE_DIST);
            if(config.with_rw_idx){
                //fwd_idx.second.occur.Sort();
                for(long i=0; i < graph_n; i++){
                    int source = i;
                    double residual = fwd_idx_residue[source];
                    total_residual+=residual;
                    unsigned long num_s_rw = ceil(residual/check_rsum*num_random_walk);
                    total_RW+=num_s_rw;
                    double a_s = residual/check_rsum*num_random_walk/num_s_rw;

                    double ppr_incre = a_s*check_rsum/num_random_walk;
                    
                    num_total_rw += num_s_rw;
                    
                    //for each source node, get rand walk destinations from previously generated idx or online rand walks
                    if(num_s_rw > rw_idx_info[source].second){ //if we need more destinations than that in idx, rand walk online
                        for(unsigned long k=0; k<rw_idx_info[source].second; k++){
                            int des = rw_idx_all[this_worker_number][rw_idx_info[source].first + k];
                            ppr[des] += ppr_incre;
                        }
                        num_hit_idx += rw_idx_info[source].second;

                        for(unsigned long j=0; j < num_s_rw-rw_idx_info[source].second; j++){ //rand walk online
                            int des = random_walk_class(source);
                            ppr[des] += ppr_incre;
                        }
                        extra_random_walk+=num_s_rw-rw_idx_info[source].second;
                    }
                    else{ // using previously generated idx is enough
                        for(unsigned long k=0; k<num_s_rw; k++){
                            int des = rw_idx_all[this_worker_number][rw_idx_info[source].first + k];
                            ppr[des] += ppr_incre;
                        }
                        num_hit_idx += num_s_rw;
                    }
                }
                printf("check extra RW: %d\n", extra_random_walk);
                printf("check total RW: %d\n", total_RW);
                printf("check total Residual: %.12f\n", total_residual);
            }
            else{ //rand walk online
                for(long i=0; i < graph_n; i++){
                    int source = i;
                    double residual = fwd_idx_residue[source];
                    unsigned long num_s_rw = ceil(residual/check_rsum*num_random_walk);
                    double a_s = residual/check_rsum*num_random_walk/num_s_rw;

                    double ppr_incre = a_s*check_rsum/num_random_walk;
                    
                    num_total_rw += num_s_rw;
                    for(unsigned long j=0; j<num_s_rw; j++){
                        int des = random_walk_class(source);
                        ppr[des] += ppr_incre;
                    }
                }
            }
        }
    }

    void agenda_query_lazy_dynamic_original_class(int v,  double theta ){
        INFO(v);
        double rsum = 1.0;
        double temp_eps=config.epsilon;
        this_worker_epsilon=config.epsilon*theta;
        this_worker_omega = (2+this_worker_epsilon)*log(2/config.pfail)/config.delta/this_worker_epsilon/this_worker_epsilon;
        
        {
            //Timer timer1(FORA_QUERY);
            //Timer timer(FWD_LU);
            double push_time_start=omp_get_wtime();
            forward_local_update_linear_class(v, rsum, config.rmax); //forward propagation, obtain reserve and residual
            double push_time_end=omp_get_wtime();
            printf("Push Time: %.12f\n", push_time_end-push_time_start);
        }

        this_worker_epsilon=temp_eps;
        
        {
            //Timer timer(REBUILD_INDEX);	
            double index_time_start=omp_get_wtime();
            lazy_update_fwdidx_class(theta);
            double index_time_end=omp_get_wtime();
            printf("Lazy Index Time: %.12f\n", index_time_end-index_time_start);
        }
        
        //Timer timer(FORA_QUERY);
        // compute_ppr_with_fwdidx(graph);
        
        {
            double ppr_time_start=omp_get_wtime();
            compute_ppr_with_fwdidx_class(rsum);
            double ppr_time_end=omp_get_wtime();
            printf("PPR Index Time: %.12f\n", ppr_time_end-ppr_time_start);
        }

    }


    void agenda_query_lazy_dynamic_error_push_class(int v,  double theta , int update_number, int error_index){
        INFO(v);
        double rsum = 1.0;
        double temp_eps=config.epsilon;
        this_worker_epsilon=config.epsilon*theta;
        this_worker_omega = (2+this_worker_epsilon)*log(2/config.pfail)/config.delta/this_worker_epsilon/this_worker_epsilon;
        double error_Shuffling=0;
        {
            //Timer timer1(FORA_QUERY);
            //Timer timer(FWD_LU);
            double push_time_start=omp_get_wtime();
            
            double error_Shuffling_bound=config.epsilon/graph.getNumOfVertices()*(1-theta)*config.beta_PFS;
            //double inaccuracy_bound=error_Shuffling_bound/(config.rmax*graph.getNumOfVertices());
            double inaccuracy_bound=0;
            printf("check inaccuracy_bound:%.12f\n", inaccuracy_bound);
            forward_local_update_linear_error_push_class(v, rsum, config.rmax, update_number, error_index, error_Shuffling, inaccuracy_bound, error_Shuffling_bound); //forward propagation, obtain reserve and residual
            double push_time_end=omp_get_wtime();
            printf("Error-Push Time: %.12f\n", push_time_end-push_time_start);
        }

        this_worker_epsilon=temp_eps;
        
        {
            //Timer timer(REBUILD_INDEX);	
            double index_time_start=omp_get_wtime();
            lazy_update_fwdidx_class_shuffling(theta);
            double index_time_end=omp_get_wtime();
            printf("Lazy Index-Extra-Error Time: %.12f\n", index_time_end-index_time_start);
        }
        
        //Timer timer(FORA_QUERY);
        // compute_ppr_with_fwdidx(graph);
        
        {
            double ppr_time_start=omp_get_wtime();
            compute_ppr_with_fwdidx_class(rsum);
            double ppr_time_end=omp_get_wtime();
            printf("PPR Index Time: %.12f\n", ppr_time_end-ppr_time_start);
        }

    }


    
//-------------------------------------------------------------------------------------------------------------
};
#endif
