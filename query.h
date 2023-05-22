
#ifndef FORA_QUERY_H
#define FORA_QUERY_H

#include "algo.h"
#include "graph.h"
#include "newGraph.h"
#include "heap.h"
#include "config.h"
#include "build.h"
#include "omp.h"
#include "agenda_class.h"
#include "MPMCQueue.h"
#include <iostream>
#include <boost/filesystem.hpp>

//#define CHECK_PPR_VALUES 1
//#define CHECK_TOP_K_PPR 1
#define PRINT_PRECISION_FOR_DIF_K 1
// std::mutex mtx;
class fora_query_topk_with_bound;
int calc_hop(Graph graph, int u, int v){
    int hop=1;
    bool find_flag=false;
	
	cout<<u<<"  "<<v<<endl;
	if(u==v)
		return 0;
    
    vector<int> list;
    
    list.push_back(u);
    
    unsigned long i=0;
    while(hop<10){
		i=0;
        vector<int> new_list;
		int length=list.size();
		//cout<<list.size()<<endl;
        while(i<length){
            int p=list[i];
            for( int next : graph.g[p] ){
                if(next==v){
                    return hop;
                }else{
                    new_list.push_back(next);
                }
            }			
			i++;
        }
		
		list=new_list;
        hop++;
    }
    return hop;
}


void montecarlo_query(int v, const Graph& graph){
    Timer timer(MC_QUERY);

    rw_counter.clean();
    ppr.reset_zero_values();

    {
        Timer tm(RONDOM_WALK);
        num_total_rw += config.omega;
        for(unsigned long i=0; i<config.omega; i++){
            int destination = random_walk(v, graph);
            if(!rw_counter.exist(destination))
                rw_counter.insert(destination, 1);
            else
                rw_counter[destination] += 1;
        }
    }

    int node_id;
    for(long i=0; i<rw_counter.occur.m_num; i++){
        node_id = rw_counter.occur[i];
        ppr[node_id] = rw_counter[node_id]*1.0/config.omega;
    }

#ifdef CHECK_PPR_VALUES
    display_ppr();
#endif
}

void montecarlo_query_topk(int v, const Graph& graph){
    Timer timer(0);

    rw_counter.clean();
    ppr.clean();

    {
        Timer tm(RONDOM_WALK);
        num_total_rw += config.omega;
        for(unsigned long i=0; i<config.omega; i++){
            int destination = random_walk(v, graph);
            if(!rw_counter.exist(destination))
                rw_counter.insert(destination, 1);
            else
                rw_counter[destination] += 1;
        }
    }

    int node_id;
    for(long i=0; i<rw_counter.occur.m_num; i++){
        node_id = rw_counter.occur[i];
        if(rw_counter.occur[i]>0)
            ppr.insert( node_id, rw_counter[node_id]*1.0/config.omega );
    }
}

void bippr_query(int v, const Graph& graph){
    Timer timer(BIPPR_QUERY);

    rw_counter.clean();
    ppr.reset_zero_values();

    {
        Timer tm(RONDOM_WALK);
        num_total_rw += config.omega;
        INFO(config.omega);
        for(unsigned long i=0; i<config.omega; i++){
            int destination = random_walk(v, graph); 
            if(!rw_counter.exist(destination))
                rw_counter.insert(destination, 1);
            else
                rw_counter[destination] += 1;
        }
    }

    INFO(config.rmax);
    if(config.rmax < 1.0){
        Timer tm(BWD_LU);
        for(long i=0; i<graph.n; i++){
            reverse_local_update_linear(i, graph);
            // if(backresult.first[v] ==0 && backresult.second.size()==0){
            if( (!bwd_idx.first.exist(v)||0==bwd_idx.first[v]) &&  0==bwd_idx.second.occur.m_num){
                continue;
            }
            ppr[i] += bwd_idx.first[v];
            // for(auto residue: backresult.second){
            for(long j=0; j<bwd_idx.second.occur.m_num; j++){
                // ppr[i]+=counts[residue.first]*1.0/config.omega*residue.second;
                int nodeid = bwd_idx.second.occur[j];
                double residual = bwd_idx.second[nodeid];
                int occur;
                if(!rw_counter.exist(nodeid))
                    occur = 0;
                else
                    occur = rw_counter[nodeid]; 

                ppr[i] += occur*1.0/config.omega*residual;
            }
        }
    }else{
        int node_id;
        for(long i=0; i<rw_counter.occur.m_num; i++){
            node_id = rw_counter.occur[i];
            ppr[node_id] = rw_counter[node_id]*1.0/config.omega;
        }
    }
#ifdef CHECK_PPR_VALUES
    display_ppr();
#endif
}

void bippr_query_topk(int v, const Graph& graph){
    Timer timer(0);

    ppr.clean();
    rw_counter.clean();

    {
        Timer tm(RONDOM_WALK);
        num_total_rw += config.omega;
        for(unsigned long i=0; i<config.omega; i++){
            int destination = random_walk(v, graph);
            if(rw_counter.notexist(destination)){
                rw_counter.insert(destination, 1);
            }
            else{
                rw_counter[destination] += 1;
            }
        }
    }

    if(config.rmax < 1.0){
        Timer tm(BWD_LU);
        for(int i=0; i<graph.n; i++){
            reverse_local_update_linear(i, graph);
            if( (!bwd_idx.first.exist(v)||0==bwd_idx.first[v]) &&  0==bwd_idx.second.occur.m_num){
                continue;
            }

            if( bwd_idx.first.exist(v) && bwd_idx.first[v]>0 )
                ppr.insert(i, bwd_idx.first[v]);

            for(long j=0; j<bwd_idx.second.occur.m_num; j++){
                int nodeid = bwd_idx.second.occur[j];
                double residual = bwd_idx.second[nodeid];
                int occur;
                if(!rw_counter.exist(nodeid)){
                    occur = 0;
                }
                else{
                    occur = rw_counter[nodeid]; 
                }

                if(occur>0){
                    if(!ppr.exist(i)){
                        ppr.insert( i, occur*residual/config.omega );
                    }
                    else{
                        ppr[i] += occur*residual/config.omega;
                    }
                }
            }
        }
    }
    else{
        int node_id;
        for(long i=0; i<rw_counter.occur.m_num; i++){
            node_id = rw_counter.occur[i];
            if(rw_counter[node_id]>0){
                if(!ppr.exist(node_id)){
                    ppr.insert( node_id, rw_counter[node_id]*1.0/config.omega );
                }
                else{
                    ppr[node_id] = rw_counter[node_id]*1.0/config.omega;
                }
            }
        }
    }
}

void hubppr_query(int s, const Graph& graph){
    Timer timer(HUBPPR_QUERY);

    ppr.reset_zero_values();

    {
        Timer tm(RONDOM_WALK);
        fwd_with_hub_oracle(graph, s);
        count_hub_dest();
        INFO("finish fwd work", hub_counter.occur.m_num, rw_counter.occur.m_num);
    }

    {
        Timer tm(BWD_LU);
        for(int t=0; t<graph.n; t++){
            bwd_with_hub_oracle(graph, t);
            // reverse_local_update_linear(t, graph);
            if( (bwd_idx.first.notexist(s) || 0==bwd_idx.first[s]) && 0==bwd_idx.second.occur.m_num ){
                continue;
            }

            if(rw_counter.occur.m_num < bwd_idx.second.occur.m_num){ //iterate on smaller-size list
                for (int i=0; i<rw_counter.occur.m_num; i++) {
                    int node = rw_counter.occur[i];
                    if (bwd_idx.second.exist(node)) {
                        ppr[t] += bwd_idx.second[node]*rw_counter[node];
                    }
                }
            }
            else{
                for (int i=0; i<bwd_idx.second.occur.m_num; i++) {
                    int node = bwd_idx.second.occur[i];
                    if (rw_counter.exist(node)) {
                        ppr[t] += rw_counter[node]*bwd_idx.second[node];
                    }
                }
            }
            ppr[t]=ppr[t]/config.omega;
            if(bwd_idx.first.exist(s))
                ppr[t] += bwd_idx.first[s];
        }
    }

#ifdef CHECK_PPR_VALUES
    display_ppr();
#endif
}

void compute_ppr_with_reserve(){
    ppr.clean();
    int node_id;
    double reserve;
    for(long i=0; i< fwd_idx.first.occur.m_num; i++){
        node_id = fwd_idx.first.occur[i];
        reserve = fwd_idx.first[ node_id ];
        if(reserve)
            ppr.insert(node_id, reserve);
    }
}

bool err_cmp(const pair<int,double> a,const pair<int,double> b){
	return a.second > b.second;
}

void lazy_update_fwdidx(const Graph& graph, double theta){
    if(config.no_rebuild)
        return;
	vector< pair<int,double> > error_idx;
	double rsum=0;
	double errsum=0;
	double inaccsum=0;
	// for(long i=0; i<fwd_idx.first.occur.m_num; i++){
    //     int node_id = fwd_idx.first.occur[i];
    for(long i=0; i<fwd_idx.second.occur.m_num; i++){
        int node_id = fwd_idx.second.occur[i];
        double reserve = fwd_idx.first[ node_id ];
		double residue = fwd_idx.second[ node_id ];
		if(residue*(1-inacc_idx[node_id])>0){
			error_idx.push_back(make_pair(node_id,residue*(1-inacc_idx[node_id])));
			rsum+=residue;
			errsum+=residue*(1-inacc_idx[node_id]);
		}
    }
	sort(error_idx.begin(), error_idx.end(), err_cmp);
	long i=0;
	//double errbound=config.epsilon/graph.n/2/rsum*(1-theta);
	double errbound=config.epsilon/graph.n*(1-theta);
    INFO(errsum);
    INFO(errbound);
	while(errsum>errbound){
		//cout<<i<<" : "<<errsum<<"--"<<error_idx[i].second<<endl;;
		update_idx(graph,error_idx[i].first);
		inacc_idx[error_idx[i].first]=1;
		errsum-=error_idx[i].second;
		i++;
	}
	//INFO(i);
}

void lazy_update_fwdidx(const NewGraph& graph, double theta){
    if(config.no_rebuild)
        return;
	vector< pair<int,double> > error_idx;
	double rsum=0;
	double errsum=0;
	double inaccsum=0;
	// for(long i=0; i<fwd_idx.first.occur.m_num; i++){
    //     int node_id = fwd_idx.first.occur[i];
    for(long i=0; i<fwd_idx.second.occur.m_num; i++){
        int node_id = fwd_idx.second.occur[i];
        double reserve = fwd_idx.first[ node_id ];
		double residue = fwd_idx.second[ node_id ];
		if(residue*(1-inacc_idx[node_id])>0){
			error_idx.emplace_back(make_pair(node_id,residue*(1-inacc_idx[node_id])));
			rsum+=residue;
			errsum+=residue*(1-inacc_idx[node_id]);
		}
    }
	sort(error_idx.begin(), error_idx.end(), err_cmp);
	long i=0;
	//double errbound=config.epsilon/graph.n/2/rsum*(1-theta);
	double errbound=config.epsilon/graph.getNumOfVertices()*(1-theta);
    INFO(errsum);
    INFO(errbound);
	while(errsum>errbound){
		//cout<<i<<" : "<<errsum<<"--"<<error_idx[i].second<<endl;;
		update_idx(graph,error_idx[i].first);
		inacc_idx[error_idx[i].first]=1;
		errsum-=error_idx[i].second;
		i++;
	}
	//INFO(i);
}

void lazy_update_fwdidx_one_hop(const Graph& graph, double theta, int v){
	vector< pair<int,double> > error_idx;
	double rsum=0;
	double errsum=0;
	double inaccsum=0;
	for(long i=0; i<fwd_idx.first.occur.m_num; i++){
        int node_id = fwd_idx.first.occur[i];
        double reserve = fwd_idx.first[ node_id ];
		double residue = fwd_idx.second[ node_id ];
		if(residue*(1-inacc_idx[node_id])>0){
			error_idx.push_back(make_pair(node_id,residue*(1-inacc_idx[node_id])));
			rsum+=residue;
			errsum+=residue*(1-inacc_idx[node_id]);
		}
    }
	sort(error_idx.begin(), error_idx.end(), err_cmp);
	long i=0;
	double errbound=config.epsilon/(config.alpha*(1-config.alpha)/graph.g[v].size())*(1-theta);
	

	
	while(errsum>errbound){
		cout<<i<<" : "<<errsum<<"--"<<error_idx[i].second<<endl;;
		update_idx(graph,error_idx[i].first);
		inacc_idx[error_idx[i].first]=1;
		errsum-=error_idx[i].second;
		i++;
	}
	//INFO(i);
}

void compute_ppr_with_fwdidx_hybrid(const Graph& graph, double check_rsum, vector<int> &rw_idx_given, 
vector< pair<unsigned long long, unsigned long> > &rw_idx_info_given){
    ppr.reset_zero_values();
	INFO("compute_ppr_with_fwdidx_hybrid");
    int node_id;
    double reserve;
	double sum=0;
	//INFO(fwd_idx.second.occur.m_num);
    for(long i=0; i< fwd_idx.first.occur.m_num; i++){
        node_id = fwd_idx.first.occur[i];
        reserve = fwd_idx.first[ node_id ];
        ppr[node_id] = reserve;
    }
	
    // INFO("rsum is:", check_rsum);
    if(check_rsum == 0.0)
        return;

    unsigned long long num_random_walk = config.omega*check_rsum;
    //num_total_rw += num_random_walk;

    {
        Timer timer(RONDOM_WALK); //both rand-walk and source distribution are related with num_random_walk
        //Timer tm(SOURCE_DIST);
        if(config.with_rw_idx){
            fwd_idx.second.occur.Sort();
            for(long i=0; i < fwd_idx.second.occur.m_num; i++){
                int source = fwd_idx.second.occur[i];
                double residual = fwd_idx.second[source];
                unsigned long num_s_rw = ceil(residual/check_rsum*num_random_walk);
                double a_s = residual/check_rsum*num_random_walk/num_s_rw;

                double ppr_incre = a_s*check_rsum/num_random_walk;
                
                num_total_rw += num_s_rw;
				rw_count += num_s_rw;
                
                //for each source node, get rand walk destinations from previously generated idx or online rand walks
                if(num_s_rw > rw_idx_info_given[source].second){ //if we need more destinations than that in idx, rand walk online
                    for(unsigned long k=0; k<rw_idx_info_given[source].second; k++){
                        int des = rw_idx_given[rw_idx_info_given[source].first + k];
                        ppr[des] += ppr_incre;
                    }
                    num_hit_idx += rw_idx_info_given[source].second;

                    for(unsigned long j=0; j < num_s_rw-rw_idx_info_given[source].second; j++){ //rand walk online
                        int des = random_walk(source, graph);
                        ppr[des] += ppr_incre;
                    }
                }else{ // using previously generated idx is enough
                    for(unsigned long k=0; k<num_s_rw; k++){
                        int des = rw_idx_given[rw_idx_info_given[source].first + k];
                        ppr[des] += ppr_incre;
                    }
                    num_hit_idx += num_s_rw;
                }
            }
        }
        else{ //rand walk online
            for(long i=0; i < fwd_idx.second.occur.m_num; i++){
                int source = fwd_idx.second.occur[i];
                double residual = fwd_idx.second[source];
                unsigned long num_s_rw = ceil(residual/check_rsum*num_random_walk);
                double a_s = residual/check_rsum*num_random_walk/num_s_rw;

                double ppr_incre = a_s*check_rsum/num_random_walk;
                
                num_total_rw += num_s_rw;
				rw_count += num_s_rw;
                for(unsigned long j=0; j<num_s_rw; j++){
                    int des = random_walk(source, graph);
                    ppr[des] += ppr_incre;
                }
            }
        }
    }
}


void compute_ppr_with_fwdidx(const Graph& graph, double check_rsum){
	
	if(config.algo==FORA_AND_BATON){
		if(config.with_baton)
			compute_ppr_with_fwdidx_hybrid(graph, check_rsum, rw_idx_baton, rw_idx_info_baton);
		else
			compute_ppr_with_fwdidx_hybrid(graph, check_rsum, rw_idx_fora, rw_idx_info_fora);
		return;
	}
    ppr.reset_zero_values();

    int node_id;
    double reserve;
	double sum=0;
	//INFO(fwd_idx.second.occur.m_num);
    for(long i=0; i< fwd_idx.first.occur.m_num; i++){
        node_id = fwd_idx.first.occur[i];
        reserve = fwd_idx.first[ node_id ];
        ppr[node_id] = reserve;
    }
	
    // INFO("rsum is:", check_rsum);
    if(check_rsum == 0.0)
        return;

    unsigned long long num_random_walk = config.omega*check_rsum;
    //num_total_rw += num_random_walk;

    {
        Timer timer(RONDOM_WALK); //both rand-walk and source distribution are related with num_random_walk
        //Timer tm(SOURCE_DIST);
        if(config.with_rw_idx){
            fwd_idx.second.occur.Sort();
            for(long i=0; i < fwd_idx.second.occur.m_num; i++){
                int source = fwd_idx.second.occur[i];
                double residual = fwd_idx.second[source];
                unsigned long num_s_rw = ceil(residual/check_rsum*num_random_walk);
                double a_s = residual/check_rsum*num_random_walk/num_s_rw;

                double ppr_incre = a_s*check_rsum/num_random_walk;
                
                num_total_rw += num_s_rw;
				rw_count += num_s_rw;
                
                //for each source node, get rand walk destinations from previously generated idx or online rand walks
                if(num_s_rw > rw_idx_info[source].second){ //if we need more destinations than that in idx, rand walk online
                    for(unsigned long k=0; k<rw_idx_info[source].second; k++){
                        int des;
                        if(config.alter_idx == 0)
                            des = rw_idx[rw_idx_info[source].first + k];
                        else
                            des = rw_idx_alter[rw_idx_info[source].first + k].back();
                        ppr[des] += ppr_incre;
                    }
                    num_hit_idx += rw_idx_info[source].second;

                    for(unsigned long j=0; j < num_s_rw-rw_idx_info[source].second; j++){ //rand walk online
                        int des = random_walk(source, graph);
                        ppr[des] += ppr_incre;
                    }
                }else{ // using previously generated idx is enough
                    for(unsigned long k=0; k<num_s_rw; k++){
                        int des;
                        if(config.alter_idx == 0)
                            des = rw_idx[rw_idx_info[source].first + k];
                        else
                            des = rw_idx_alter[rw_idx_info[source].first + k].back();
                        ppr[des] += ppr_incre;
                    }
                    num_hit_idx += num_s_rw;
                }
            }
        }
        else{ //rand walk online
            for(long i=0; i < fwd_idx.second.occur.m_num; i++){
                int source = fwd_idx.second.occur[i];
                double residual = fwd_idx.second[source];
                unsigned long num_s_rw = ceil(residual/check_rsum*num_random_walk);
                double a_s = residual/check_rsum*num_random_walk/num_s_rw;

                double ppr_incre = a_s*check_rsum/num_random_walk;
                
                num_total_rw += num_s_rw;
				rw_count += num_s_rw;
                for(unsigned long j=0; j<num_s_rw; j++){
                    int des = random_walk(source, graph);
                    ppr[des] += ppr_incre;
                }
            }
        }
    }
}

void compute_ppr_with_fwdidx(const NewGraph& graph, double check_rsum){
    ppr.reset_zero_values();

    int node_id;
    double reserve;
    for(long i=0; i< fwd_idx.first.occur.m_num; i++){
        node_id = fwd_idx.first.occur[i];
        reserve = fwd_idx.first[ node_id ];
        ppr[node_id] = reserve;
    }
    //INFO(ppr.occur.m_num);

    // INFO("rsum is:", check_rsum);
    if(check_rsum == 0.0)
        return;

    unsigned long long num_random_walk = config.omega*check_rsum;
    //INFO(num_random_walk);
    //num_total_rw += num_random_walk;

    {
        Timer timer(RONDOM_WALK); //both rand-walk and source distribution are related with num_random_walk
        //Timer tm(SOURCE_DIST);
        if(config.with_rw_idx){
            fwd_idx.second.occur.Sort();
            for(long i=0; i < fwd_idx.second.occur.m_num; i++){
                int source = fwd_idx.second.occur[i];
                double residual = fwd_idx.second[source];
                unsigned long num_s_rw = ceil(residual/check_rsum*num_random_walk);
                double a_s = residual/check_rsum*num_random_walk/num_s_rw;

                double ppr_incre = a_s*check_rsum/num_random_walk;
                
                num_total_rw += num_s_rw;
                
                //for each source node, get rand walk destinations from previously generated idx or online rand walks
                if(num_s_rw > rw_idx_info[source].second){ //if we need more destinations than that in idx, rand walk online
                    for(unsigned long k=0; k<rw_idx_info[source].second; k++){
                        int des = rw_idx[rw_idx_info[source].first + k];
                        ppr[des] += ppr_incre;
                    }
                    num_hit_idx += rw_idx_info[source].second;

                    for(unsigned long j=0; j < num_s_rw-rw_idx_info[source].second; j++){ //rand walk online
                        int des = random_walk(source, graph);
                        ppr[des] += ppr_incre;
                    }
                }else{ // using previously generated idx is enough
                    for(unsigned long k=0; k<num_s_rw; k++){
                        int des = rw_idx[rw_idx_info[source].first + k];
                        ppr[des] += ppr_incre;
                    }
                    num_hit_idx += num_s_rw;
                }
            }
        }
        else{ //rand walk online
            for(long i=0; i < fwd_idx.second.occur.m_num; i++){
                int source = fwd_idx.second.occur[i];
                double residual = fwd_idx.second[source];
                unsigned long num_s_rw = ceil(residual/check_rsum*num_random_walk);
                double a_s = residual/check_rsum*num_random_walk/num_s_rw;

                double ppr_incre = a_s*check_rsum/num_random_walk;
                
                num_total_rw += num_s_rw;
                for(unsigned long j=0; j<num_s_rw; j++){
                    int des = random_walk(source, graph);
                    ppr[des] += ppr_incre;
                }
            }
        }
    }
}

void compute_ppr_with_fwdidx_opt(const NewGraph& graph, double check_rsum){
    ppr.reset_zero_values();

    int node_id;
    double reserve;
    for(long i=0; i< fwd_idx.first.occur.m_num; i++){
        node_id = fwd_idx.first.occur[i];
        reserve = fwd_idx.first[ node_id ];
        ppr[node_id] = reserve;
    }
    //INFO(ppr.occur.m_num);

    // INFO("rsum is:", check_rsum);
    if(check_rsum == 0.0)
        return;

    check_rsum*=(1-config.alpha);
    unsigned long long num_random_walk = config.omega*check_rsum;
    //INFO(num_random_walk);
    //num_total_rw += num_random_walk;

    {
        Timer timer(RONDOM_WALK); //both rand-walk and source distribution are related with num_random_walk
        //Timer tm(SOURCE_DIST);
        if(config.with_rw_idx){
            fwd_idx.second.occur.Sort();
            for(long i=0; i < fwd_idx.second.occur.m_num; i++){
                int source = fwd_idx.second.occur[i];
                //double residual = fwd_idx.second[source];
                if(!fwd_idx.second.exist(source)) continue;
                ppr[source]+=fwd_idx.second[source]*config.alpha;
                double residual = fwd_idx.second[source]*(1-config.alpha);


                unsigned long num_s_rw = ceil(residual/check_rsum*num_random_walk);
                double a_s = residual/check_rsum*num_random_walk/num_s_rw;

                double ppr_incre = a_s*check_rsum/num_random_walk;
                
                num_total_rw += num_s_rw;
                
                //for each source node, get rand walk destinations from previously generated idx or online rand walks
                if(num_s_rw > rw_idx_info[source].second){ //if we need more destinations than that in idx, rand walk online
                    for(unsigned long k=0; k<rw_idx_info[source].second; k++){
                        int des = rw_idx[rw_idx_info[source].first + k];
                        ppr[des] += ppr_incre;
                    }
                    num_hit_idx += rw_idx_info[source].second;

                    for(unsigned long j=0; j < num_s_rw-rw_idx_info[source].second; j++){ //rand walk online
                        int des = random_walk(source, graph);
                        ppr[des] += ppr_incre;
                    }
                }else{ // using previously generated idx is enough
                    for(unsigned long k=0; k<num_s_rw; k++){
                        int des = rw_idx[rw_idx_info[source].first + k];
                        ppr[des] += ppr_incre;
                    }
                    num_hit_idx += num_s_rw;
                }
            }
        }
        else{ //rand walk online
            for(long i=0; i < fwd_idx.second.occur.m_num; i++){
                int source = fwd_idx.second.occur[i];
                if(!fwd_idx.second.exist(source)) continue;
                ppr[source]+=fwd_idx.second[source]*config.alpha;
                double residual = fwd_idx.second[source]*(1-config.alpha);
                unsigned long num_s_rw = ceil(residual/check_rsum*num_random_walk);
                double a_s = residual/check_rsum*num_random_walk/num_s_rw;

                double ppr_incre = a_s*check_rsum/num_random_walk;
                num_total_rw += num_s_rw;
                for(unsigned long j=0; j<num_s_rw; j++){
                    int des = random_walk(source, graph);
                    ppr[des] += ppr_incre;
                }
            }
        }
    }
}

void compute_ppr_with_fwdidx_opt(const Graph& graph, double check_rsum){
    ppr.reset_zero_values();

    int node_id;
    double reserve;
    for(long i=0; i< fwd_idx.first.occur.m_num; i++){
        node_id = fwd_idx.first.occur[i];
        reserve = fwd_idx.first[ node_id ];
        ppr[node_id] = reserve;
    }

    // INFO("rsum is:", check_rsum);
    if(check_rsum == 0.0)
        return;

    check_rsum*=(1-config.alpha);
    unsigned long long num_random_walk = config.omega*check_rsum;
    // INFO(num_random_walk);
    //num_total_rw += num_random_walk;

    {
        Timer timer(RONDOM_WALK); //both rand-walk and source distribution are related with num_random_walk
        //Timer tm(SOURCE_DIST);
        if(config.with_rw_idx){
            fwd_idx.second.occur.Sort();
            for(long i=0; i < fwd_idx.second.occur.m_num; i++){
                int source = fwd_idx.second.occur[i];
                //double residual = fwd_idx.second[source];
                if(!fwd_idx.second.exist(source)) continue;
                ppr[source]+=fwd_idx.second[source]*config.alpha;
                double residual = fwd_idx.second[source]*(1-config.alpha);


                unsigned long num_s_rw = ceil(residual/check_rsum*num_random_walk);
                double a_s = residual/check_rsum*num_random_walk/num_s_rw;

                double ppr_incre = a_s*check_rsum/num_random_walk;
                
                num_total_rw += num_s_rw;
                
                //for each source node, get rand walk destinations from previously generated idx or online rand walks
                if(num_s_rw > rw_idx_info[source].second){ //if we need more destinations than that in idx, rand walk online
                    for(unsigned long k=0; k<rw_idx_info[source].second; k++){
                        int des = rw_idx[rw_idx_info[source].first + k];
                        ppr[des] += ppr_incre;
                    }
                    num_hit_idx += rw_idx_info[source].second;

                    for(unsigned long j=0; j < num_s_rw-rw_idx_info[source].second; j++){ //rand walk online
                        int des = random_walk_no_zero_hop(source, graph);
                        ppr[des] += ppr_incre;
                    }
                }else{ // using previously generated idx is enough
                    for(unsigned long k=0; k<num_s_rw; k++){
                        int des = rw_idx[rw_idx_info[source].first + k];
                        ppr[des] += ppr_incre;
                    }
                    num_hit_idx += num_s_rw;
                }
            }
        }
        else{ //rand walk online
            for(long i=0; i < fwd_idx.second.occur.m_num; i++){
                int source = fwd_idx.second.occur[i];
                if(!fwd_idx.second.exist(source)) continue;
                ppr[source]+=fwd_idx.second[source]*config.alpha;
                double residual = fwd_idx.second[source]*(1-config.alpha);
                unsigned long num_s_rw = ceil(residual/check_rsum*num_random_walk);
                double a_s = residual/check_rsum*num_random_walk/num_s_rw;

                double ppr_incre = a_s*check_rsum/num_random_walk;
                num_total_rw += num_s_rw;
                for(unsigned long j=0; j<num_s_rw; j++){
                    int des = random_walk_no_zero_hop(source, graph);
                    ppr[des] += ppr_incre;
                }
            }
        }
    }
}

void compute_ppr_with_fwdidx_topk_with_bound_hybrid(const Graph& graph, double check_rsum,
vector<int> &rw_idx_given, 
vector< pair<unsigned long long, unsigned long> > &rw_idx_info_given, double theta=1.0){
    compute_ppr_with_reserve();
	INFO("compute_ppr_with_fwdidx_topk_with_bound_hybrid");
    if(check_rsum == 0.0)
        return;

    long num_random_walk = config.omega*check_rsum;
		INFO(config.omega,check_rsum);
    long real_num_rand_walk=0;
	rw_counter.reset_zero_values();
    {
        Timer timer(RONDOM_WALK); //both rand-walk and source distribution are related with num_random_walk

        //Timer tm(SOURCE_DIST);
        if(config.with_rw_idx){ //rand walk with previously generated idx
            fwd_idx.second.occur.Sort();
            //for each source node, get rand walk destinations from previously generated idx or online rand walks
            for(long i=0; i < fwd_idx.second.occur.m_num; i++){
                int source = fwd_idx.second.occur[i];
				//cout<<"source: "<<source<<endl;
                double residual = fwd_idx.second[source];
                long num_s_rw = ceil(residual/check_rsum*num_random_walk);
				
                double a_s = residual/check_rsum*num_random_walk/num_s_rw;
								
                double ppr_incre = a_s*check_rsum/num_random_walk;

                real_num_rand_walk += num_s_rw;

                long num_used_idx = 0;
                bool source_cnt_exist = rw_counter.exist(source);
                if( source_cnt_exist ){
                    num_used_idx = rw_counter[source];
					//if(num_used_idx>0)
						//continue;
				}
				/*
				if( graph.g[source].size() == 0 ){
					if(ppr.exist(source))
						ppr[source] += ppr_incre*num_s_rw;
					else
						ppr.insert(source, ppr_incre*num_s_rw);
					if(source==24980){
								cout<<24980<<":\t"<<ppr[source]<<endl;
								cout<<ppr_incre<<":\t"<<num_s_rw<<endl;
					}
					continue;
				}*/
				
				//cout<<"num_used_idx: "<<num_used_idx<<endl;
				num_total_rw += num_s_rw;
				
                long num_remaining_idx = rw_idx_info_given[source].second-num_used_idx;
				//cout<<"num_remaining_idx: "<<num_remaining_idx<<endl;
                if(num_s_rw <= num_remaining_idx){
                    // using previously generated idx is enough
                    long k=0;
                    for(; k<num_remaining_idx; k++){
											rw_count++;
                        if( k < num_s_rw){
                            int des = rw_idx_given[rw_idx_info_given[source].first + k];
                            if(ppr.exist(des))
                                ppr[des] += ppr_incre;
                            else
                                ppr.insert(des, ppr_incre);
                        }else
                            break;
                    }
                    if(source_cnt_exist){
                        rw_counter[source] += k;
                    }
                    else{
                        rw_counter.insert(source, k);
                    }

                    num_hit_idx += k;
					//cout<<"num_hit_idx: "<<num_hit_idx<<endl;
                }else{
                    //we need more destinations than that in idx, rand walk online
                    for(long k=0; k<num_remaining_idx; k++){
                        int des = rw_idx_given[ rw_idx_info_given[source].first + k ];
                        if(!ppr.exist(des))
                            ppr.insert(des, ppr_incre);
                        else
                            ppr[des] += ppr_incre;
                    }
                    num_hit_idx += num_remaining_idx;

                    if(!source_cnt_exist){
                        rw_counter.insert( source, num_remaining_idx );
                    }
                    else{
                        rw_counter[source] += num_remaining_idx;
                    }

                    for(long j=0; j < num_s_rw-num_remaining_idx; j++){ //rand walk online
                        int des = random_walk(source, graph);
                        if(!ppr.exist(des))
                            ppr.insert(des, ppr_incre);
                        else 
                            ppr[des] += ppr_incre;
                    }
                }
            }
        }
        else{ //rand walk online
			double residue_sum=0;
			/*
			long rw_sum;
			long rw_sum_1=0;
			long rw_sum_2=0;
			double residue_sum_1=0;
			 
			for(long i=0; i<fwd_idx.second.occur.m_num; i++){
				long id =fwd_idx.second.occur[i];
				residue_sum += fwd_idx.second[id];
				rw_sum_1+=ceil(fwd_idx.second[id]/check_rsum*num_random_walk);
			}
			rw_sum = ceil(residue_sum/check_rsum*num_random_walk);
			cout<<"res: "<<residue_sum<<endl;
			cout<<"rw: "<<rw_sum<<endl;
			 */
			
            for(int i=0; i < fwd_idx.second.occur.m_num; i++){
                int source = fwd_idx.second.occur[i];
                double residual = fwd_idx.second[source];
				//residue_sum_1 += fwd_idx.second[source];
                long num_s_rw = ceil(residual/check_rsum*num_random_walk);
                double a_s = residual/check_rsum*num_random_walk/num_s_rw;
				//rw_sum_2+=num_s_rw;

                real_num_rand_walk += num_s_rw;

                double ppr_incre = a_s*check_rsum/num_random_walk;
                for(long j=0; j<num_s_rw; j++){
                    int des = random_walk(source, graph);
                    if(!ppr.exist(des))
                        ppr.insert(des, ppr_incre);
                    else
                        ppr[des] += ppr_incre;

                }
            }
			/*
			cout<<"rs1: "<<residue_sum_1<<endl;
			cout<<"rw1:"<<rw_sum_1<<endl;
			cout<<"rw2:"<<rw_sum_2<<endl;
			cout<<"real_rw"<<real_num_rand_walk<<endl;
			*/
        }
		/*
		for(long i=0; i<ppr.occur.m_num; i++){
						long id = ppr.occur[i];
						cout<<"id: "<<id<<"  rw_value: "<<ppr[id]<<endl;
		}
		 */
    }

    if(config.delta < threshold)
		if(theta==1.0)
			set_ppr_bounds(graph, check_rsum, real_num_rand_walk);
		else
			set_ppr_bounds_dynamic(graph, check_rsum, real_num_rand_walk, theta);
    else{
        zero_ppr_upper_bound = calculate_lambda(check_rsum,  config.pfail, zero_ppr_upper_bound, real_num_rand_walk, theta);
    }
}

void compute_ppr_with_fwdidx_topk(const Graph& graph, double check_rsum){
    // ppr.clean();
    // // ppr.reset_zero_values();

    // int node_id;
    // double reserve;
    // for(long i=0; i< fwd_idx.first.occur.m_num; i++){
    //     node_id = fwd_idx.first.occur[i];
    //     reserve = fwd_idx.first[ node_id ];
    //     ppr.insert(node_id, reserve);
    //     // ppr[node_id] = reserve;
    // }
    compute_ppr_with_reserve();

    // INFO("rsum is:", check_rsum);
    if(check_rsum == 0.0)
        return;

    check_rsum*= (1-config.alpha);
    unsigned long long num_random_walk = config.omega*check_rsum;
    // INFO(num_random_walk);
    //num_total_rw += num_random_walk;
    {
        Timer timer(RONDOM_WALK); //both rand-walk and source distribution are related with num_random_walk
        //Timer tm(SOURCE_DIST);
        int source;
        double residual;
        unsigned long num_s_rw;
        double a_s;
        double ppr_incre;
        unsigned long num_used_idx;
        unsigned long num_remaining_idx;
        int des;
        //INFO(num_random_walk, fwd_idx.second.occur.m_num);
        if(config.with_rw_idx){ //rand walk with previously generated idx
            fwd_idx.second.occur.Sort();
            //for each source node, get rand walk destinations from previously generated idx or online rand walks
            for(long i=0; i < fwd_idx.second.occur.m_num; i++){
                source = fwd_idx.second.occur[i];
                residual = fwd_idx.second[source];
                if(ppr.exist(source)){
                    ppr[source] += residual*config.alpha;
                }else{
                    ppr.insert(source, residual*config.alpha);
                }

                residual*=(1-config.alpha);
                num_s_rw = ceil(residual*config.omega);
                a_s = residual*config.omega/num_s_rw;

                ppr_incre = a_s/config.omega;

                num_total_rw += num_s_rw;

                num_used_idx = rw_counter[source];
                num_remaining_idx = rw_idx_info[source].second;
                
                if(num_s_rw <= num_remaining_idx){
                    // using previously generated idx is enough
                    for(unsigned long k=0; k<num_s_rw; k++){
                        des = rw_idx[ rw_idx_info[source].first+ num_used_idx + k ];
                        if(!ppr.exist(des))
                            ppr.insert(des, ppr_incre);
                        else
                            ppr[des] += ppr_incre;
                    }

                    rw_counter[source] = num_used_idx + num_s_rw;

                    num_hit_idx += num_s_rw;
                }
                else{
                    //INFO(num_s_rw,num_remaining_idx);
                    //we need more destinations than that in idx, rand walk online
                    for(unsigned long k=0; k<num_remaining_idx; k++){
                        des = rw_idx[ rw_idx_info[source].first + num_used_idx + k ];
                        if(!ppr.exist(des))
                            ppr.insert(des, ppr_incre);
                        else
                            ppr[des] += ppr_incre;
                    }

                    num_hit_idx += num_remaining_idx;
                    rw_counter[source] = num_used_idx + num_remaining_idx;

                    for(unsigned long j=0; j < num_s_rw-num_remaining_idx; j++){ //rand walk online
                        des = random_walk_no_zero_hop(source, graph);
                        if(!ppr.exist(des))
                            ppr.insert(des, ppr_incre);
                        else
                            ppr[des] += ppr_incre;
                    }
                }
            }
        }
        else{ //rand walk online
            for(long i=0; i < fwd_idx.second.occur.m_num; i++){
                source = fwd_idx.second.occur[i];
                residual = fwd_idx.second[source];
                num_s_rw = ceil(residual*config.omega);
                a_s = residual*config.omega/num_s_rw;

                ppr_incre = a_s/config.omega;
                num_total_rw += num_s_rw;

                for(unsigned long j=0; j<num_s_rw; j++){
                    des = random_walk(source, graph);
                    if(!ppr.exist(des))
                        ppr.insert(des, ppr_incre);
                    else
                        ppr[des] += ppr_incre;
                }
            }
        }
    }

}

void compute_ppr_with_fwdidx_topk_with_bound(const Graph& graph, double check_rsum, double theta=1.0){
	if(config.algo==FORA_AND_BATON){
		if(config.with_baton)
			compute_ppr_with_fwdidx_topk_with_bound_hybrid(graph, check_rsum, rw_idx_baton, rw_idx_info_baton, theta);
		else
			compute_ppr_with_fwdidx_topk_with_bound_hybrid(graph, check_rsum, rw_idx_fora, rw_idx_info_fora, theta);
		return;
	}
    compute_ppr_with_reserve();
    if(check_rsum == 0.0)
        return;

    long num_random_walk = config.omega*check_rsum;
    long real_num_rand_walk=0;
	rw_counter.reset_zero_values();
    {
        Timer timer(RONDOM_WALK); //both rand-walk and source distribution are related with num_random_walk

        //Timer tm(SOURCE_DIST);
        if(config.with_rw_idx){ //rand walk with previously generated idx
            fwd_idx.second.occur.Sort();
            //for each source node, get rand walk destinations from previously generated idx or online rand walks
            for(long i=0; i < fwd_idx.second.occur.m_num; i++){
                int source = fwd_idx.second.occur[i];
				//cout<<"source: "<<source<<endl;
                double residual = fwd_idx.second[source];
                long num_s_rw = ceil(residual/check_rsum*num_random_walk);
				
                double a_s = residual/check_rsum*num_random_walk/num_s_rw;
				

                double ppr_incre = a_s*check_rsum/num_random_walk;

               
                real_num_rand_walk += num_s_rw;

                long num_used_idx = 0;
                bool source_cnt_exist = rw_counter.exist(source);
                if( source_cnt_exist ){
                    num_used_idx = rw_counter[source];
					//if(num_used_idx>0)
						//continue;
				}
				
				/*
				if( graph.g[source].size() == 0 ){
					if(ppr.exist(source))
						ppr[source] += ppr_incre*num_s_rw;
					else
						ppr.insert(source, ppr_incre*num_s_rw);
					if(source==24980){
								cout<<24980<<":\t"<<ppr[source]<<endl;
								cout<<ppr_incre<<":\t"<<num_s_rw<<endl;
					}
					continue;
				}*/
				
				//cout<<"num_used_idx: "<<num_used_idx<<endl;
				num_total_rw += num_s_rw;
				
                long num_remaining_idx = rw_idx_info[source].second-num_used_idx;
				//cout<<"num_remaining_idx: "<<num_remaining_idx<<endl;
				
                
                if(num_s_rw <= num_remaining_idx){
                    // using previously generated idx is enough
                    long k=0;
                    for(; k<num_remaining_idx; k++){
											rw_count++;
                        if( k < num_s_rw){
                            int des = rw_idx[rw_idx_info[source].first + k];
                            if(ppr.exist(des))
                                ppr[des] += ppr_incre;
                            else
                                ppr.insert(des, ppr_incre);
                        }else
                            break;
                    }
                    if(source_cnt_exist){
                        rw_counter[source] += k;
                    }
                    else{
                        rw_counter.insert(source, k);
                    }

                    num_hit_idx += k;
					//cout<<"num_hit_idx: "<<num_hit_idx<<endl;
                }else{
                    //we need more destinations than that in idx, rand walk online

                    for(long k=0; k<num_remaining_idx; k++){
                        int des = rw_idx[ rw_idx_info[source].first + k ];
                        if(!ppr.exist(des))
                            ppr.insert(des, ppr_incre);
                        else
                            ppr[des] += ppr_incre;
                    }
                    num_hit_idx += num_remaining_idx;

                    if(!source_cnt_exist){
                        rw_counter.insert( source, num_remaining_idx );
                    }
                    else{
                        rw_counter[source] += num_remaining_idx;
                    }

                    for(long j=0; j < num_s_rw-num_remaining_idx; j++){ //rand walk online
                        int des = random_walk(source, graph);
                        if(!ppr.exist(des))
                            ppr.insert(des, ppr_incre);
                        else 
                            ppr[des] += ppr_incre;
                    }
                }
            }
        }
        else{ //rand walk online
			double residue_sum=0;
			/*
			long rw_sum;
			long rw_sum_1=0;
			long rw_sum_2=0;
			double residue_sum_1=0;
			 
			for(long i=0; i<fwd_idx.second.occur.m_num; i++){
				long id =fwd_idx.second.occur[i];
				residue_sum += fwd_idx.second[id];
				rw_sum_1+=ceil(fwd_idx.second[id]/check_rsum*num_random_walk);
			}
			rw_sum = ceil(residue_sum/check_rsum*num_random_walk);
			cout<<"res: "<<residue_sum<<endl;
			cout<<"rw: "<<rw_sum<<endl;
			 */
			
            for(int i=0; i < fwd_idx.second.occur.m_num; i++){
                int source = fwd_idx.second.occur[i];
                double residual = fwd_idx.second[source];
				//residue_sum_1 += fwd_idx.second[source];
                long num_s_rw = ceil(residual/check_rsum*num_random_walk);
                double a_s = residual/check_rsum*num_random_walk/num_s_rw;
				//rw_sum_2+=num_s_rw;

                real_num_rand_walk += num_s_rw;

                double ppr_incre = a_s*check_rsum/num_random_walk;
                for(long j=0; j<num_s_rw; j++){
                    int des = random_walk(source, graph);
                    if(!ppr.exist(des))
                        ppr.insert(des, ppr_incre);
                    else
                        ppr[des] += ppr_incre;

                }
            }
			/*
			cout<<"rs1: "<<residue_sum_1<<endl;
			cout<<"rw1:"<<rw_sum_1<<endl;
			cout<<"rw2:"<<rw_sum_2<<endl;
			cout<<"real_rw"<<real_num_rand_walk<<endl;
			*/
        }
		/*
		for(long i=0; i<ppr.occur.m_num; i++){
						long id = ppr.occur[i];
						cout<<"id: "<<id<<"  rw_value: "<<ppr[id]<<endl;
		}
		 */
    }

    if(config.delta < threshold)
		if(theta==1.0)
			set_ppr_bounds(graph, check_rsum, real_num_rand_walk);
		else
			set_ppr_bounds_dynamic(graph, check_rsum, real_num_rand_walk, theta);
    else{
        zero_ppr_upper_bound = calculate_lambda(check_rsum,  config.pfail, zero_ppr_upper_bound, real_num_rand_walk, theta);
    }
}




void compute_ppr_with_fwdidx_topk_with_bound_baton(const Graph& graph, double check_rsum){

    compute_ppr_with_reserve();

    if(check_rsum == 0.0)
        return;

    long num_random_walk = config.omega*check_rsum;
    long real_num_rand_walk=0;
	rw_counter.reset_zero_values();
    {
        Timer timer(RONDOM_WALK); //both rand-walk and source distribution are related with num_random_walk

        //Timer tm(SOURCE_DIST);
        if(config.with_rw_idx){ //rand walk with previously generated idx
            fwd_idx.second.occur.Sort();
			//cout<<fwd_idx.second.occur.m_num<<endl;
            //for each source node, get rand walk destinations from previously generated idx or online rand walks
            for(long i=0; i < fwd_idx.second.occur.m_num; i++){
                int source = fwd_idx.second.occur[i];
				//cout<<"source: "<<source<<endl;
                double residual = fwd_idx.second[source];
				if(residual==0)
					continue;
                long num_s_rw = ceil(residual/check_rsum*num_random_walk);
				
                double a_s = residual/check_rsum*num_random_walk/num_s_rw;

                double ppr_incre = a_s*check_rsum/num_random_walk;

               
                real_num_rand_walk += num_s_rw;

                long num_used_idx = 0;
                bool source_cnt_exist = rw_counter.exist(source);
                if( source_cnt_exist ){
                    num_used_idx = rw_counter[source];
					//if(num_used_idx>0)
						//continue;
				}
				if( graph.g[source].size() == 0 ){
					if(ppr.exist(source))
						ppr[source] += ppr_incre*num_s_rw;
					else
						ppr.insert(source, ppr_incre*num_s_rw);
					continue;
				}
                long num_reuse_idx = 0;
				
				//cout<<"++++++++++++++++"<<endl;
				//cout<<"source: "<<source<<endl;
				
				num_reuse_idx=used_idx[source];

				//cout<<"num_used_idx: "<<num_used_idx<<endl;
				num_total_rw += num_s_rw;
                long num_remaining_idx = rw_idx_info[source].second;
                long num_noreuse_idx = num_s_rw - num_reuse_idx;
				//cout<<"num_remaining_idx: "<<num_remaining_idx<<endl;
				num_hit_idx += num_reuse_idx + num_noreuse_idx;
                //cout<<"ppr_incre"<<ppr_incre<<endl;
				//cout<<num_s_rw<<endl;
				//cout<<"-"<<num_reuse_idx<<"-   -"<<num_noreuse_idx<<"-"<<endl;
                if(num_s_rw <= num_remaining_idx){
					
                    // using previously generated idx is enough
					if(num_noreuse_idx>0){
						long k=0;
						//cout<<num_noreuse_idx<<endl;
						//cout<<num_reuse_idx<<endl;
						for(; k<num_noreuse_idx; k++){
							rw_count++;
							int des = rw_idx[rw_idx_info[source].first + k + num_reuse_idx];
							/*
							if(reuse_idx.exist(des))
								reuse_idx[des] ++;
							else
								reuse_idx.insert(des, 1);
								 */
							reuse_idx_vector[des]++;
						}	
						used_idx[source]=num_s_rw;
					}else{
						long k=0;
						for(; k>num_noreuse_idx; k--){
							rw_count++;
							int des = rw_idx[rw_idx_info[source].first + k + num_reuse_idx - 1];
							reuse_idx_vector[des]--;
						}
						used_idx[source]=num_s_rw;
						
					}
					//cout<<"num_hit_idx: "<<num_hit_idx<<endl;
                }else{
                    //we need more destinations than that in idx, rand walk online
					
					cout<<"num_s_rw: "<<num_s_rw<<endl;
					cout<<"idx: "<<rw_idx_info[source].second<<endl;
					cout<<"source: "<<source<<endl;
					cout<<"dout: "<<graph.g[source].size()<<endl;
					
                    for(long k=0; k<num_remaining_idx; k++){
                        int des = rw_idx[ rw_idx_info[source].first + k ];
                        if(!ppr.exist(des))
                            ppr.insert(des, ppr_incre);
                        else
                            ppr[des] += ppr_incre;
                    }
                    num_hit_idx += num_remaining_idx;

                    if(!source_cnt_exist){
                        rw_counter.insert( source, num_remaining_idx );
                    }
                    else{
                        rw_counter[source] += num_remaining_idx;
                    }

                    for(long j=0; j < num_s_rw-num_remaining_idx; j++){ //rand walk online
                        int des = random_walk(source, graph);
                        if(!ppr.exist(des))
                            ppr.insert(des, ppr_incre);
                        else 
                            ppr[des] += ppr_incre;
                    }
                }
            }
        }
        else{ //rand walk online
            for(long i=0; i < fwd_idx.second.occur.m_num; i++){
                int source = fwd_idx.second.occur[i];
                double residual = fwd_idx.second[source];
                long num_s_rw = ceil(residual/check_rsum*num_random_walk);
                double a_s = residual/check_rsum*num_random_walk/num_s_rw;

                real_num_rand_walk += num_s_rw;

                double ppr_incre = a_s*check_rsum/num_random_walk;
                for(long j=0; j<num_s_rw; j++){
                    int des = random_walk(source, graph);
                    if(!ppr.exist(des))
                        ppr.insert(des, ppr_incre);
                    else
                        ppr[des] += ppr_incre;

                }
            }
        }
		Timer tm(20);
		double ppr_incre=check_rsum/num_random_walk;
		long length=reuse_idx_vector.size();
		//cout<<reuse_idx.occur.m_num<<endl;
		for(long i=0; i < length; i++){
			if(reuse_idx_vector[i]>0){
				if(ppr.exist(i))
					ppr[i] += ppr_incre*reuse_idx_vector[i];
				else
					ppr.insert(i, ppr_incre*reuse_idx_vector[i]);
			}
		}
		
	
    }

    if(config.delta < threshold)
        set_ppr_bounds(graph, check_rsum, real_num_rand_walk);
    else{
        zero_ppr_upper_bound = calculate_lambda(check_rsum,  config.pfail, zero_ppr_upper_bound, real_num_rand_walk);
    }
}

void resacc_query(int v, const Graph& graph){
    Timer timer(FORA_QUERY);
	//INFO(v);
    double rsum = 0.0;
	double r_max_hop=0.1;
	double r_max_f=0.1;
	int k_hops=2;
	int vert = graph.n;
	double* resacc_residual = new double[vert];
	double* resacc_ppr = new double[vert];
	int* hops_from_source = new int[vert];
	for(int i = 0; i < vert; i++){
		resacc_ppr[i] = 0.0;
		resacc_residual[i] = 0.0;
		hops_from_source[i] = numeric_limits<int>::max();
	}
	resacc_residual[v] = 1.0;
	hops_from_source[v] = 0;
	unordered_set<int> kHopSet;
	unordered_set<int> k1HopLayer;
    {
        Timer timer(FWD_LU);
		k1HopLayer = kHopFWD(v, k_hops, hops_from_source, kHopSet, k1HopLayer, r_max_hop, resacc_ppr, resacc_residual, graph);
		//for(int i = 0; i < vert; i++)
			//rsum+=resacc_residual[i];
			rsum=1;
		r_max_f=config.epsilon * r_max_f / sqrt( (graph.m) * 3.0 * log(2.0 * graph.n) * graph.n);
        forward_local_update_resacc(v, graph, rsum, config.rmax, resacc_residual, resacc_ppr, k1HopLayer); //forward propagation, obtain reserve and residual
    }

    // compute_ppr_with_fwdidx(graph);
    compute_ppr_with_fwdidx(graph, rsum);

#ifdef CHECK_PPR_VALUES
    display_ppr();
#endif
}

double total_rsum = 0.0;
double random_walk_time = 0.0000004;
double random_walk_index_time = random_walk_time/140;
double previous_rmax = 0;

double estimated_random_walk_cost(double rsum, double rmax){
    double estimated_random_walk_cost = 0.0;
    if(!config.with_rw_idx){
        estimated_random_walk_cost = config.omega*rsum*(1-config.alpha)*random_walk_time;
    }else{
        if(rmax >= config.rmax){
            estimated_random_walk_cost = config.omega*rsum*(1-config.alpha)*random_walk_time;
        }else{
            estimated_random_walk_cost = config.omega*rsum*(1-config.alpha)*random_walk_index_time;
        }
    }
    INFO(rmax, config.rmax, estimated_random_walk_cost);
    return estimated_random_walk_cost;
}

void fora_query_basic(int v, const Graph& graph){
    Timer timer(FORA_QUERY);
	//INFO(v);
    double rsum = 1.0;
    if(config.balanced){
        static vector<int> forward_from;
        forward_from.clear();
        forward_from.reserve(graph.n);
        forward_from.push_back(v);

        fwd_idx.first.clean();  //reserve
        fwd_idx.second.clean();  //residual
        fwd_idx.second.insert( v, rsum );

        const static double min_delta = 1.0 / graph.n;

        const static double lowest_delta_rmax = config.opt?config.epsilon*sqrt(min_delta/3/graph.m/log(2/config.pfail))/(1-config.alpha):config.epsilon*sqrt(min_delta/3/graph.m/log(2/config.pfail));
        double used_time = 0;
        double rmax = 0;
        rmax = config.rmax*8;
        double random_walk_cost = 0;
        if(graph.g[v].size()>0){
            while(estimated_random_walk_cost(rsum, rmax)> used_time){
                INFO(config.omega*rsum*random_walk_time, used_time);
                std::chrono::steady_clock::time_point startTime = std::chrono::steady_clock::now();
                forward_local_update_linear_topk( v, graph, rsum, rmax, lowest_delta_rmax, forward_from ); 
                auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - startTime).count();
                INFO(rsum);
                used_time +=duration/TIMES_PER_SEC;
                double used_time_this_iteration = duration/TIMES_PER_SEC;
                INFO(used_time_this_iteration);
                rmax /=2;
            }
            rmax*=2;
            INFO("Adpaitve total forward push time: ", used_time);
            INFO(config.rmax, rmax, config.rmax/rmax);
            //count_ratio[config.rmax/rmax]++;
        }else{
            forward_local_update_linear(v, graph, rsum, config.rmax);
        }
    }else{
        Timer timer(FWD_LU);
        forward_local_update_linear(v, graph, rsum, config.rmax); //forward propagation, obtain reserve and residual
    }

    if(config.opt){
        compute_ppr_with_fwdidx_opt(graph, rsum);
        total_rsum+= rsum*(1-config.alpha);
    }else{
        compute_ppr_with_fwdidx(graph, rsum);
        total_rsum+= rsum*(1-config.alpha);
    }

#ifdef CHECK_PPR_VALUES
    display_ppr();
#endif
}

void fora_query_dynamic(int v, const Graph& graph, double sigma){
    Timer timer(FORA_QUERY);
    double rsum = 1.0;
	double temp_eps=config.epsilon;
	ASSERT(sigma<1);
	config.epsilon=config.epsilon*(1-sigma);
	config.omega = (2+config.epsilon)*log(2/config.pfail)/config.delta/config.epsilon/config.epsilon;
	//cout<<"111"<<endl;
    {
        Timer timer(FWD_LU);
        forward_local_update_linear(v, graph, rsum, config.rmax); //forward propagation, obtain reserve and residual
    }
	//cout<<"222"<<endl;
	config.epsilon=temp_eps;
    // compute_ppr_with_fwdidx(graph);
    compute_ppr_with_fwdidx(graph, rsum);

#ifdef CHECK_PPR_VALUES
    display_ppr();
#endif
}

void fora_query_lazy_dynamic(int v, const Graph& graph, double theta){
	INFO(v);
    double rsum = 1.0;
	double temp_eps=config.epsilon;
	config.epsilon=config.epsilon*theta;
	config.omega = (2+config.epsilon)*log(2/config.pfail)/config.delta/config.epsilon/config.epsilon;
	
    {
		Timer timer1(FORA_QUERY);
        Timer timer(FWD_LU);
        forward_local_update_linear(v, graph, rsum, config.rmax); //forward propagation, obtain reserve and residual
    }
	config.epsilon=temp_eps;
	
	{
		Timer timer(REBUILD_INDEX);	
		lazy_update_fwdidx(graph, theta);
	}
	
	Timer timer(FORA_QUERY);
    // compute_ppr_with_fwdidx(graph);
	
    if(config.opt){
        compute_ppr_with_fwdidx_opt(graph, rsum);
    }else{
        compute_ppr_with_fwdidx(graph, rsum);
    }

#ifdef CHECK_PPR_VALUES
    display_ppr();
#endif

}

void fora_query_lazy_dynamic(int v, const NewGraph& newGraph, double theta ){
	INFO(v);
    double rsum = 1.0;
	double temp_eps=config.epsilon;
	config.epsilon=config.epsilon*theta;
	config.omega = (2+config.epsilon)*log(2/config.pfail)/config.delta/config.epsilon/config.epsilon;
	
    {
		Timer timer1(FORA_QUERY);
        Timer timer(FWD_LU);
        forward_local_update_linear(v, newGraph, rsum, config.rmax); //forward propagation, obtain reserve and residual
    }
	config.epsilon=temp_eps;
	
	{
		Timer timer(REBUILD_INDEX);	
		lazy_update_fwdidx(newGraph, theta);
	}
	
	Timer timer(FORA_QUERY);
    // compute_ppr_with_fwdidx(graph);
	
    if(config.opt){
        compute_ppr_with_fwdidx_opt(newGraph, rsum);
    }else{
        compute_ppr_with_fwdidx(newGraph, rsum);
    }

#ifdef CHECK_PPR_VALUES
    display_ppr();
#endif

}

void one_hop_query_dynamic(int v, const Graph& graph, double theta, bool is_lazy=false){
	
    double rsum = 1.0;
	double temp_eps=config.epsilon;
	config.epsilon=config.epsilon*theta;
	config.omega = (2+config.epsilon)*log(2/config.pfail)/(config.alpha*(1-config.alpha)/graph.g[v].size())/config.epsilon/config.epsilon;
	
    {
		Timer timer1(FORA_QUERY);
        Timer timer(FWD_LU);
        forward_local_update_linear(v, graph, rsum, config.rmax); //forward propagation, obtain reserve and residual
    }
	config.epsilon=temp_eps;
	
	if(is_lazy){
		Timer timer(REBUILD_INDEX);	
		lazy_update_fwdidx_one_hop(graph, theta, v);
	}
	
	Timer timer(FORA_QUERY);
    // compute_ppr_with_fwdidx(graph);
	
    compute_ppr_with_fwdidx(graph, rsum);

#ifdef CHECK_PPR_VALUES
    display_ppr();
#endif

}

void fora_query_topk_new(int v, const Graph& graph ){
    Timer timer(0);
    const static double min_delta = 1.0 / graph.n;
    if(config.k ==0) config.k = 500;
    const static double init_delta = 1.0/config.k/10;//(1.0-config.ppr_decay_alpha)/pow(500, config.ppr_decay_alpha) / pow(graph.n, 1-config.ppr_decay_alpha);
    const static double new_pfail = 1.0 / graph.n / graph.n;

    config.pfail = new_pfail;  // log(1/pfail) -> log(1*n/pfail)
    config.delta = init_delta;

    const static double lowest_delta_rmax = config.epsilon*sqrt(min_delta/3/graph.m/log(2/new_pfail));

    double rsum = 1.0;

    static vector<int> forward_from;
    forward_from.clear();
    forward_from.reserve(graph.n);
    forward_from.push_back(v);

    fwd_idx.first.clean();  //reserve
    fwd_idx.second.clean();  //residual
    fwd_idx.second.insert( v, rsum );

    

    if(config.with_rw_idx)
        rw_counter.reset_zero_values(); //store the pointers of each node's used rand-walk idxs 

    // for delta: try value from 1/4 to 1/n
    while( config.delta >= min_delta ){
        fora_topk_setting(graph.n, graph.m);
        num_iter_topk++;
        {
            Timer timer(FWD_LU);
            //INFO(config.rmax, graph.m*config.rmax, config.omega);
            if(graph.g[v].size()==0){
                rsum = 0.0;
                fwd_idx.first.insert(v, 1);
                compute_ppr_with_reserve();
                return;
            }else{
                forward_local_update_linear_topk( v, graph, rsum, config.rmax, lowest_delta_rmax, forward_from ); //forward propagation, obtain reserve and residual
            }
        }

        //i_destination_count.clean();
        //compute_ppr_with_fwdidx_new(graph, rsum);
        //compute_ppr_with_fwdidx_topk(graph, rsum);
        compute_ppr_with_fwdidx_topk(graph, rsum);

        
        {
            double kth_ppr_score = kth_ppr();

            //topk_ppr();

            //double kth_ppr_score = topk_pprs[config.k-1].second;
            if( kth_ppr_score >= (1+config.epsilon)*config.delta || config.delta <= min_delta ){  // once k-th ppr value in top-k list >= (1+epsilon)*delta, terminate
                INFO(kth_ppr_score, config.delta, rsum);
                break;
            }
            else{
                /*int j=0;
                for(; j<config.k; j++){
                    //INFO(topk_pprs[j].second, (1+config.epsilon)*config.delta);
                    if(topk_pprs[j].second<(1+config.epsilon)*config.delta)
                        break;
                }
                INFO("Our current accurate top-j", j);*/
                config.delta = max( min_delta, config.delta/4.0 );  // otherwise, reduce delta to delta/4
            }
        }
    }
}

void fora_query_topk_with_bound(int v, const Graph& graph){

    Timer tm_0(0);

    const static double min_delta = 1.0 / graph.n;
    const static double init_delta = 1.0 / config.k;
    threshold = (1.0-config.ppr_decay_alpha)/pow(500, config.ppr_decay_alpha) / pow(graph.n, 1-config.ppr_decay_alpha);
	//cout<<"threshold: "<<threshold<<endl;

    const static double new_pfail = 1.0 / graph.n / graph.n/log(graph.n);

    config.pfail = new_pfail;  // log(1/pfail) -> log(1*n/pfail)
    config.delta = init_delta;

    double lowest_delta_rmax = config.epsilon*sqrt(min_delta/3/graph.m/log(2/new_pfail));
	if(config.with_baton == true)
		lowest_delta_rmax = config.beta/(config.omega*config.alpha);

    double rsum = 1.0;

    static vector<int> forward_from;
    forward_from.clear();
    forward_from.reserve(graph.n);
    forward_from.push_back(v);

    fwd_idx.first.clean();  //reserve
    fwd_idx.second.clean();  //residual
    fwd_idx.second.insert( v, rsum );

    zero_ppr_upper_bound = 1.0;

    if(config.with_rw_idx)
        rw_counter.reset_zero_values(); //store the pointers of each node's used rand-walk idxs 

    // for delta: try value from 1/4 to 1/n
    int iteration = 0;
    upper_bounds.reset_one_values();
    lower_bounds.reset_zero_values();
	
	ppr_rw.clean();
	//reuse_idx.clean();
	reuse_idx_vector.reserve(graph.n);
	reuse_idx_vector.assign(graph.n, 0);
	used_idx.assign(graph.n, 0);
	current_cycle=1000;
    while( config.delta >= min_delta ){
		cycle+=1;
		current_cycle++;
		//Timer tm(current_cycle);
		//cout<<"++++++++++++++++++++++++++++++++++"<<endl;
        fora_setting(graph.n, graph.m);
        num_iter_topk++;

        {
            Timer timer(FWD_LU);
            forward_local_update_linear_topk( v, graph, rsum, config.rmax, lowest_delta_rmax, forward_from ); //forward propagation, obtain reserve and residual
        }
		if(config.reuse == true)
			compute_ppr_with_fwdidx_topk_with_bound_baton(graph, rsum);
		else
			compute_ppr_with_fwdidx_topk_with_bound(graph, rsum);
        if(if_stop() || config.delta <= min_delta){
            break;
        }else
            config.delta = max( min_delta, config.delta/config.n );  // otherwise, reduce delta to delta/2
    }
	cout<<"Id: "<<v<<"  Delta: "<<config.delta<<endl;
	//display_setting();
}

iMap<int> updated_pprs;
void hubppr_query_topk_martingale(int s, const Graph& graph) {
    unsigned long long the_omega = 2*config.rmax*log(2*config.k/config.pfail)/config.epsilon/config.epsilon/config.delta;
    static double bwd_cost_div = 1.0*graph.m/graph.n/config.alpha;

    static double min_ppr = 1.0/graph.n;
    static double new_pfail = config.pfail/2.0/graph.n/log2(1.0*graph.n*config.alpha*graph.n*graph.n);
    static double pfail_star = log(new_pfail/2);

    static std::vector<bool> target_flag(graph.n);
    static std::vector<double> m_omega(graph.n);
    static vector<vector<int>> node_targets(graph.n);
    static double cur_rmax=1;

    // rw_counter.clean();
    for(int t=0; t<graph.n; t++){
        map_lower_bounds[t].second = 0;//min_ppr;
        upper_bounds[t] = 1.0;
        target_flag[t] = true;
        m_omega[t]=0;
    }

    int num_iter = 1;
    int target_size=graph.n;
    if(cur_rmax>config.rmax){
        cur_rmax=config.rmax;
        for(int t=0; t<graph.n; t++){
            if(target_flag[t]==false)
                continue;
            reverse_local_update_topk(s, t, reserve_maps[t], cur_rmax, residual_maps[t], graph);
            for(const auto &p: residual_maps[t]){
                node_targets[p.first].push_back(t);
            }
        }
    }
    while( target_size > config.k && num_iter<=64 ){ //2^num_iter <= 2^64 since 2^64 is the largest unsigned integer here
        unsigned long long num_rw = pow(2, num_iter);
        rw_counter.clean();
        generate_accumulated_fwd_randwalk(s, graph, num_rw);
        updated_pprs.clean();
        // update m_omega
        {
            for(int x=0; x<rw_counter.occur.m_num; x++){
                int node = rw_counter.occur[x];
                for(const int t: node_targets[node]){
                    if(target_flag[t]==false)
                        continue;
                    m_omega[t] += rw_counter[node]*residual_maps[t][node];
                    if(!updated_pprs.exist(t))
                        updated_pprs.insert(t, 1);
                }
            }
        }

        double b = (2*num_rw-1)*pow(cur_rmax/2.0, 2);
        double lambda = sqrt(pow(cur_rmax*pfail_star/3, 2) - 2*b*pfail_star) - cur_rmax*pfail_star/3;
        {
            for(int i=0; i<updated_pprs.occur.m_num; i++){
                int t = updated_pprs.occur[i];
                if( target_flag[t]==false )
                    continue;

                double reserve = 0;
                if(reserve_maps[t].find(s)!=reserve_maps[t].end()){
                    reserve = reserve_maps[t][s];
                }
                set_martingale_bound(lambda, 2*num_rw-1, t, reserve, cur_rmax, pfail_star, min_ppr, m_omega[t]);
            }
        }

        topk_pprs.clear();
        topk_pprs.resize(config.k);
        partial_sort_copy(map_lower_bounds.begin(), map_lower_bounds.end(), topk_pprs.begin(), topk_pprs.end(), 
            [](pair<int, double> const& l, pair<int, double> const& r){return l.second > r.second;});

        double k_bound = topk_pprs[config.k-1].second;
        if( k_bound*(1+config.epsilon) >= upper_bounds[topk_pprs[config.k-1].first] || (num_rw >= the_omega && cur_rmax <= config.rmax) ){
            break;
        }

        for(int t=0; t<graph.n; t++){
            if(target_flag[t]==true && upper_bounds[t] <= k_bound){
                target_flag[t] = false;
                target_size--;
            }
        }
        num_iter++;
    }
}

void get_topk_dynamic(int v, Graph &graph, double theta, bool if_lazy){
    min_delta = 1.0 / graph.n;
    const static double init_delta = 1.0 / config.k;
    threshold = (1.0-config.ppr_decay_alpha)/pow(500, config.ppr_decay_alpha) / pow(graph.n, 1-config.ppr_decay_alpha);
	//cout<<"threshold: "<<threshold<<endl;

    const static double new_pfail = 1.0 / graph.n / graph.n/log(graph.n);

    config.pfail = new_pfail;  // log(1/pfail) -> log(1*n/pfail)
    config.delta = init_delta;

    double lowest_delta_rmax = config.epsilon*sqrt(min_delta/3/graph.m/log(2/new_pfail));
	if(config.with_baton == true)
		lowest_delta_rmax = config.beta/(config.omega*config.alpha);

    double rsum = 1.0;

    static vector<int> forward_from;
    forward_from.clear();
    forward_from.reserve(graph.n);
    forward_from.push_back(v);

    fwd_idx.first.clean();  //reserve
    fwd_idx.second.clean();  //residual
    fwd_idx.second.insert( v, rsum );

    zero_ppr_upper_bound = 1.0;

    if(config.with_rw_idx)
        rw_counter.reset_zero_values(); //store the pointers of each node's used rand-walk idxs 

    // for delta: try value from 1/4 to 1/n
    int iteration = 0;
    upper_bounds.reset_one_values();
    lower_bounds.reset_zero_values();
	
	ppr_rw.clean();
	//reuse_idx.clean();
	reuse_idx_vector.reserve(graph.n);
	reuse_idx_vector.assign(graph.n, 0);
	used_idx.assign(graph.n, 0);
	current_cycle=1000;
    while( config.delta >= min_delta ){
		cycle+=1;
		current_cycle++;
		//Timer tm(current_cycle);
		//cout<<"++++++++++++++++++++++++++++++++++"<<endl;
        fora_setting(graph.n, graph.m);
        num_iter_topk++;

        {
            Timer timer(FWD_LU);
            forward_local_update_linear_topk( v, graph, rsum, config.rmax, lowest_delta_rmax, forward_from ); //forward propagation, obtain reserve and residual
        }
		if(config.algo == LAZYUP){
			Timer timer(REBUILD_INDEX);	
			lazy_update_fwdidx(graph, theta);			
		}
		compute_ppr_with_fwdidx_topk_with_bound(graph, rsum, theta);
        if(if_stop() || config.delta <= min_delta){
            break;
        }else
            config.delta = max( min_delta, config.delta/config.n );  // otherwise, reduce delta to delta/2
    }
	//cout<<"Id: "<<v<<"  Delta: "<<config.delta<<endl;
	topk_ppr();
}

void get_topk_dynamic_new(int v, Graph &graph, double theta, bool if_lazy){
    const static double min_delta = 1.0 / graph.n;
    
    if(config.k ==0) config.k = 500;
    const static double init_delta = 1.0/config.k/10;//(1.0-config.ppr_decay_alpha)/pow(500, config.ppr_decay_alpha) / pow(graph.n, 1-config.ppr_decay_alpha);
    const static double new_pfail = 1.0 / graph.n / graph.n;

    config.pfail = new_pfail;  // log(1/pfail) -> log(1*n/pfail)
    config.delta = init_delta;

    double lowest_delta_rmax = config.epsilon*sqrt(min_delta/3/graph.m/log(2/new_pfail));
    if(config.with_baton == true)
		lowest_delta_rmax = config.beta/(config.omega*config.alpha);

    double rsum = 1.0;

    static vector<int> forward_from;
    forward_from.clear();
    forward_from.reserve(graph.n);
    forward_from.push_back(v);

    fwd_idx.first.clean();  //reserve
    fwd_idx.second.clean();  //residual
    fwd_idx.second.insert( v, rsum );

    

    if(config.with_rw_idx)
        rw_counter.reset_zero_values(); //store the pointers of each node's used rand-walk idxs 

    // for delta: try value from 1/4 to 1/n
    config.epsilon=config.epsilon/2;
    while( config.delta >= min_delta ){
        fora_topk_setting(graph.n, graph.m);
        num_iter_topk++;
        {
            Timer timer(FWD_LU);
            //INFO(config.rmax, graph.m*config.rmax, config.omega);
            if(graph.g[v].size()==0){
                rsum = 0.0;
                fwd_idx.first.insert(v, 1);
                compute_ppr_with_reserve();
                return;
            }else{
                forward_local_update_linear_topk( v, graph, rsum, config.rmax, lowest_delta_rmax, forward_from ); //forward propagation, obtain reserve and residual
            }
        }
        if(config.algo == LAZYUP){
			Timer timer(REBUILD_INDEX);	
			lazy_update_fwdidx(graph, theta);			
		}
        //i_destination_count.clean();
        //compute_ppr_with_fwdidx_new(graph, rsum);
        //compute_ppr_with_fwdidx_topk(graph, rsum);
        compute_ppr_with_fwdidx_topk(graph, rsum);

        
        {
            double kth_ppr_score = kth_ppr();

            //topk_ppr();

            //double kth_ppr_score = topk_pprs[config.k-1].second;
            if( kth_ppr_score >= (1+config.epsilon)*config.delta || config.delta <= min_delta ){  // once k-th ppr value in top-k list >= (1+epsilon)*delta, terminate
                INFO(kth_ppr_score, config.delta, rsum);
                break;
            }
            else{
                /*int j=0;
                for(; j<config.k; j++){
                    //INFO(topk_pprs[j].second, (1+config.epsilon)*config.delta);
                    if(topk_pprs[j].second<(1+config.epsilon)*config.delta)
                        break;
                }
                INFO("Our current accurate top-j", j);*/
                config.delta = max( min_delta, config.delta/4.0 );  // otherwise, reduce delta to delta/4
            }
        }
    }
	topk_ppr();
    config.epsilon=config.epsilon*2;
}

void get_topk(int v, Graph &graph){
	config.delta = 1.0/graph.n;
	fora_setting(graph.n, graph.m);
    //display_setting();
    if(config.algo == MC){
        montecarlo_query_topk(v, graph);
        topk_ppr();
    }
    else if(config.algo == BIPPR){
        bippr_query_topk(v, graph);
        topk_ppr();
    }
    else if(config.algo == FORA||config.algo == BATON||config.algo == FORA_NO_INDEX||config.algo == FORA_AND_BATON){
        if(config.opt)
            fora_query_topk_new(v, graph);
        else
            fora_query_topk_with_bound(v, graph);
        topk_ppr();
    }
    else if(config.algo == FWDPUSH){
        Timer timer(0);
        double rsum = 1;
        
        {
            Timer timer(FWD_LU);
            forward_local_update_linear(v, graph, rsum, config.rmax);
        }
        compute_ppr_with_reserve();
        topk_ppr();
    }
    else if(config.algo == HUBPPR){
        Timer timer(0);
        hubppr_query_topk_martingale(v, graph);
    }

     // not FORA, so it's single source
     // no need to change k to run again
     // check top-k results for different k
    if(config.algo != FORA  && config.algo != HUBPPR){
        compute_precision_for_dif_k(v);
    }
    compute_precision(v);

#ifdef CHECK_TOP_K_PPR
    //vector<pair<int, double>>& exact_result = exact_topk_pprs[v];
    INFO("query node:", v);
	cout<<"i: "<<topk_pprs.size()<<endl;
    for(int i=0; i<topk_pprs.size(); i++){
		//int hop=calc_hop(graph,v,topk_pprs[i].first);
		//Hop[hop]++;
        cout << "Estimated "<<i<<"-th node: " << topk_pprs[i].first << " PPR score: " << topk_pprs[i].second << " " //<< map_lower_bounds[topk_pprs[i].first].first<< " " << map_lower_bounds[topk_pprs[i].first].second
		<<endl;//<<" hop: "<<hop<<endl;    //<<" Exact k-th node: " << exact_result[i].first << " PPR score: " << exact_result[i].second << endl;
    }
	for(int i=0; i<11; i++){
		cout<<i<< " hop count :"<<Hop[i]<<endl; 
	}
#endif
}

void fwd_power_iteration(const Graph& graph, int start, unordered_map<int, double>& map_ppr){
    //static thread_local unordered_map<int, double> map_residual;
	unordered_map<int, double> map_residual;
    map_residual[start] = 1.0;

    int num_iter=0;
    double rsum = 1.0;
    while( num_iter < config.max_iter_num ){
        num_iter++;
        INFO(num_iter, rsum, map_residual.size());
        vector< pair<int,double> > pairs(map_residual.begin(), map_residual.end());
        map_residual.clear();
        for(const auto &p: pairs){
            if(p.second > 0){
                map_ppr[p.first] += config.alpha*p.second;
                int out_deg = graph.g[p.first].size();

                double remain_residual = (1-config.alpha)*p.second;
                rsum -= config.alpha*p.second;
                if(out_deg==0){
                    map_residual[start] += remain_residual;
                }
                else{
                    double avg_push_residual = remain_residual / out_deg;
                    for (int next : graph.g[p.first]) {
                        map_residual[next] += avg_push_residual;
                    }
                }
            }
        }
        pairs.clear();
    }
    map_residual.clear();
}

void multi_power_iter(const Graph& graph, const vector<int>& source, unordered_map<int, vector<pair<int ,double>>>& map_topk_ppr ){
    static thread_local unordered_map<int, double> map_ppr;
    for(int start: source){
        fwd_power_iteration(graph, start, map_ppr);

        vector<pair<int ,double>> temp_top_ppr(config.k);
        partial_sort_copy(map_ppr.begin(), map_ppr.end(), temp_top_ppr.begin(), temp_top_ppr.end(), 
            [](pair<int, double> const& l, pair<int, double> const& r){return l.second > r.second;});
        
        map_ppr.clear();
        map_topk_ppr[start] = temp_top_ppr;
    }
}

void gen_exact_topk(const Graph& graph){
    // config.epsilon = 0.5;
    // montecarlo_setting();

    vector<int> queries;
    load_ss_query(queries);
    INFO(queries.size());
    unsigned int query_size = queries.size();
    query_size = min( query_size, config.query_size );
    INFO(query_size);

    assert(config.k < graph.n-1);
    assert(config.k > 1);
    INFO(config.k);

    split_line();
    // montecarlo_setting();

    unsigned NUM_CORES = std::thread::hardware_concurrency()-1;
    assert(NUM_CORES >= 2);

    int num_thread = min(query_size, NUM_CORES);
    int avg_queries_per_thread = query_size/num_thread;

    vector<vector<int>> source_for_all_core(num_thread);
    vector<unordered_map<int, vector<pair<int ,double>>>> ppv_for_all_core(num_thread);

    for(int tid=0; tid<num_thread; tid++){
        int s = tid*avg_queries_per_thread;
        int t = s+avg_queries_per_thread;

        if(tid==num_thread-1)
            t+=query_size%num_thread;

        for(;s<t;s++){
            // cout << s+1 <<". source node:" << queries[s] << endl;
            source_for_all_core[tid].push_back(queries[s]);
        }
    }


    {
        Timer timer(PI_QUERY);
        INFO("power itrating...");
        std::vector< std::future<void> > futures(num_thread);
        for(int tid=0; tid<num_thread; tid++){
            futures[tid] = std::async( std::launch::async, multi_power_iter, std::ref(graph), std::ref(source_for_all_core[tid]), std::ref(ppv_for_all_core[tid]) );
        }
        std::for_each( futures.begin(), futures.end(), std::mem_fn(&std::future<void>::wait));
    }

    // cout << "average iter times:" << num_iter_topk/query_size << endl;
    cout << "average generation time (s): " << Timer::used(PI_QUERY)*1.0/query_size << endl;

    INFO("combine results...");
    for(int tid=0; tid<num_thread; tid++){
        for(auto &ppv: ppv_for_all_core[tid]){
            exact_topk_pprs.insert( ppv );
        }
        ppv_for_all_core[tid].clear();
    }

    save_exact_topk_ppr();
}

void topk(Graph& graph){
    vector<int> queries;
    load_ss_query(queries);
    INFO(queries.size());
    unsigned int query_size = queries.size();
    query_size = min( query_size, config.query_size );
    int used_counter=0;

    assert(config.k < graph.n-1);
    assert(config.k > 1);
    INFO(config.k);

    split_line();
    
    load_exact_topk_ppr();

     // not FORA, so it's single source
     // no need to change k to run again
     // check top-k results for different k
    if(config.algo != FORA && config.algo != HUBPPR){
        unsigned int step = config.k/5;
        if(step > 0){
            for(unsigned int i=1; i<5; i++){
                ks.push_back(i*step);
            }
        }
        ks.push_back(config.k);
        for(auto k: ks){	
            PredResult rst(0,0,0,0,0);
            pred_results.insert(MP(k, rst));
        }
    }

    used_counter = 0; 
    if(config.algo == FORA){
        fwd_idx.first.initialize(graph.n);
        fwd_idx.second.initialize(graph.n);
        rw_counter.init_keys(graph.n);
        upper_bounds.init_keys(graph.n);
        lower_bounds.init_keys(graph.n);
        ppr.initialize(graph.n);
		ppr_rw.initialize(graph.n);
        topk_filter.initialize(graph.n);
		//reuse_idx.initialize(graph.n);
		used_idx.reserve(graph.n);
		used_idx.assign(graph.n, 0);
		reuse_idx_vector.reserve(graph.n);
		reuse_idx_vector.assign(graph.n, 0);
		
		//reuse_idx.first.initialize(graph.n);
        //reuse_idx.second.initialize(graph.n);
    }
    else if(config.algo == MC){
        rw_counter.initialize(graph.n);
        ppr.initialize(graph.n);
        montecarlo_setting();
    }
    else if(config.algo == BIPPR){
        bippr_setting(graph.n, graph.m);
        rw_counter.initialize(graph.n);
        bwd_idx.first.initialize(graph.n);
        bwd_idx.second.initialize(graph.n);
        ppr.initialize(graph.n); 
    }
    else if(config.algo == FWDPUSH){
        fwdpush_setting(graph.n, graph.m);
        fwd_idx.first.initialize(graph.n);
        fwd_idx.second.initialize(graph.n);
        ppr.initialize(graph.n);
    }
    else if(config.algo == HUBPPR){
        hubppr_topk_setting(graph.n, graph.m);
        rw_counter.initialize(graph.n);
        upper_bounds.init_keys(graph.n);
        if(config.with_rw_idx){
            hub_counter.initialize(graph.n);
            load_hubppr_oracle(graph);
        }
        residual_maps.resize(graph.n);
        reserve_maps.resize(graph.n);
        map_lower_bounds.resize(graph.n);
        for(int v=0; v<graph.n; v++){
            residual_maps[v][v]=1.0;
            map_lower_bounds[v] = MP(v, 0);
        }
        updated_pprs.initialize(graph.n);
    }

    for(int i=0; i<query_size; i++){
        //cout << i+1 <<". source node:" << queries[i] << endl;
        get_topk(queries[i], graph);
		/*
		for(int next : graph.g[queries[i]]){
			//cout<<next<<endl;
			get_topk(next, graph);
		}
		 */
        split_line();
		 
    }

    cout << "average iter times:" << num_iter_topk/query_size << endl;
    display_time_usage(used_counter, query_size);
    set_result(graph, used_counter, query_size);

     //not FORA, so it's single source
     //no need to change k to run again
     // check top-k results for different k
    if(config.algo != FORA && config.algo != HUBPPR){
        display_precision_for_dif_k();
    }
}

void query(Graph& graph){
    INFO(config.algo);
    vector<int> queries;
    load_ss_query(queries);
    unsigned int query_size = queries.size();
    query_size = min( query_size, config.query_size );
    INFO(query_size);
    int used_counter=0;

    // assert(config.rw_cost_ratio >= 0);
    // INFO(config.rw_cost_ratio); 

    assert(config.rmax_scale >= 0);
    INFO(config.rmax_scale);

    // ppr.initialize(graph.n);
    ppr.init_keys(graph.n);
    // sfmt_init_gen_rand(&sfmtSeed , 95082);

    if(config.algo == BIPPR){ //bippr
        bippr_setting(graph.n, graph.m);
        display_setting();
        used_counter = BIPPR_QUERY;

        bwd_idx.first.initialize(graph.n);
        bwd_idx.second.initialize(graph.n);

        rw_counter.initialize(graph.n);
        for(int i=0; i<query_size; i++){
            cout << i+1 <<". source node:" << queries[i] << endl;
            bippr_query(queries[i], graph);
            split_line();
        }
    }else if(config.algo == HUBPPR){
        bippr_setting(graph.n, graph.m);
        display_setting();
        used_counter = HUBPPR_QUERY;

        bwd_idx.first.initialize(graph.n);
        bwd_idx.second.initialize(graph.n);
        hub_counter.initialize(graph.n);
        rw_counter.initialize(graph.n);
        
        load_hubppr_oracle(graph);
        for(int i=0; i<query_size; i++){
            cout << i+1 <<". source node:" << queries[i] << endl;
            hubppr_query(queries[i], graph);
            split_line();
        }
    }
    else if(config.algo == FORA){ //fora
        fora_setting(graph.n, graph.m);
        display_setting();
        used_counter = FORA_QUERY;

        fwd_idx.first.initialize(graph.n);
        fwd_idx.second.initialize(graph.n);

        // if(config.multithread)
        //     vec_ppr.resize(graph.n);

        // rw_counter.initialize(graph.n);
        for(int i=0; i<query_size; i++){
            cout << i+1 <<". source node:" << queries[i] << endl;
            fora_query_basic(queries[i], graph);
            split_line();
        }
    }
	else if(config.algo == GENDA){ //fora
        fora_setting(graph.n, graph.m);
        display_setting();
        used_counter = FORA_QUERY;

        fwd_idx.first.initialize(graph.n);
        fwd_idx.second.initialize(graph.n);
		if(config.adaptive){
			config.update_size=0;
			set_optimal_beta(config,graph);
		}
		rebuild_idx(graph);
        // if(config.multithread)
        //     vec_ppr.resize(graph.n);

        // rw_counter.initialize(graph.n);
        for(int i=0; i<query_size; i++){
            cout << i+1 <<". source node:" << queries[i] << endl;
            fora_query_basic(queries[i], graph);
            split_line();
        }
    }else if(config.algo == RESACC){ //fora
        fora_setting(graph.n, graph.m);
        display_setting();
        used_counter = FORA_QUERY;

        fwd_idx.first.initialize(graph.n);
        fwd_idx.second.initialize(graph.n);

        // if(config.multithread)
        //     vec_ppr.resize(graph.n);

        // rw_counter.initialize(graph.n);
        for(int i=0; i<query_size; i++){
            cout << i+1 <<". source node:" << queries[i] << endl;
            resacc_query(queries[i], graph);
            split_line();
        }
    }else if(config.algo == MC){ //mc
        montecarlo_setting();
        display_setting();
        used_counter = MC_QUERY;

        rw_counter.initialize(graph.n);

        for(int i=0; i<query_size; i++){
            cout << i+1 <<". source node:" << queries[i] << endl;
            montecarlo_query(queries[i], graph);
            split_line();
        }
    }
    else if(config.algo == FWDPUSH){
        fwdpush_setting(graph.n, graph.m);
        display_setting();
        used_counter = FWD_LU;

        fwd_idx.first.initialize(graph.n);
        fwd_idx.second.initialize(graph.n);

        for(int i=0; i<query_size; i++){
            cout << i+1 <<". source node:" << queries[i] << endl;
            Timer timer(used_counter);
            double rsum = 1;
            forward_local_update_linear(queries[i], graph, rsum, config.rmax);
            compute_ppr_with_reserve();
            split_line();
        }
    }

    display_time_usage(used_counter, query_size);
    set_result(graph, used_counter, query_size);
}

void batch_topk(Graph& graph){
    vector<int> queries;
    load_ss_query(queries);
    INFO(queries.size());
    unsigned int query_size = queries.size();
    query_size = min( query_size, config.query_size );
    int used_counter=0;

    assert(config.k < graph.n-1);
    assert(config.k > 1);
    INFO(config.k);

    split_line();

    used_counter = 0; 
    if(config.algo == FORA){
        fwd_idx.first.initialize(graph.n);
        fwd_idx.second.initialize(graph.n);
        rw_counter.init_keys(graph.n);
        upper_bounds.init_keys(graph.n);
        lower_bounds.init_keys(graph.n);
        ppr.initialize(graph.n);
        topk_filter.initialize(graph.n);
    }
    else if(config.algo == MC){
        rw_counter.initialize(graph.n);
        ppr.initialize(graph.n);
        montecarlo_setting();
    }
    else if(config.algo == BIPPR){
        bippr_setting(graph.n, graph.m);
        rw_counter.initialize(graph.n);
        bwd_idx.first.initialize(graph.n);
        bwd_idx.second.initialize(graph.n);  
        ppr.initialize(graph.n); 
    }
    else if(config.algo == FWDPUSH){
        fwdpush_setting(graph.n, graph.m);
        fwd_idx.first.initialize(graph.n);
        fwd_idx.second.initialize(graph.n);
        ppr.initialize(graph.n);
    }
    else if(config.algo == HUBPPR){
        hubppr_topk_setting(graph.n, graph.m);
        rw_counter.initialize(graph.n); 
        upper_bounds.init_keys(graph.n);
        if(config.with_rw_idx){
            hub_counter.initialize(graph.n);
            load_hubppr_oracle(graph);
        }
        residual_maps.resize(graph.n);
        reserve_maps.resize(graph.n);
        map_lower_bounds.resize(graph.n);
        for(int v=0; v<graph.n; v++){
            residual_maps[v][v]=1.0;
            map_lower_bounds[v] = MP(v, 0);
        }
        updated_pprs.initialize(graph.n);
    }

    unsigned int step = config.k/5;
    if(step > 0){
        for(unsigned int i=1; i<5; i++){
            ks.push_back(i*step);
        }
    }
    ks.push_back(config.k);
    for(auto k: ks){
        PredResult rst(0,0,0,0,0);
        pred_results.insert(MP(k, rst));
    }

    // not FORA, so it's of single source
    // no need to change k to run again
    // check top-k results for different k
    if(config.algo != FORA && config.algo != HUBPPR ){
        for(int i=0; i<query_size; i++){
            cout << i+1 <<". source node:" << queries[i] << endl;
            get_topk(queries[i], graph);
            split_line();
        }

        display_time_usage(used_counter, query_size);
        set_result(graph, used_counter, query_size);

        display_precision_for_dif_k();
    }
    else{ // for FORA, when k is changed, run algo again
        for(unsigned int k: ks){
            config.k = k;
            INFO("========================================");
            INFO("k is set to be ", config.k);
            result.topk_recall=0;
            result.topk_precision=0;
            result.real_topk_source_count=0;
            Timer::clearAll();
            for(int i=0; i<query_size; i++){
                cout << i+1 <<". source node:" << queries[i] << endl;
                get_topk(queries[i], graph);
                split_line();
            }
            pred_results[k].topk_precision=result.topk_precision;
            pred_results[k].topk_recall=result.topk_recall;
            pred_results[k].real_topk_source_count=result.real_topk_source_count;

            cout << "k=" << k << " precision=" << result.topk_precision/result.real_topk_source_count 
                              << " recall=" << result.topk_recall/result.real_topk_source_count << endl;
            cout << "Average query time (s):"<<Timer::used(used_counter)/query_size<<endl;
            Timer::reset(used_counter);
        }

        // display_time_usage(used_counter, query_size);
        display_precision_for_dif_k();
    }
}

void generate_dynamic_workload(bool if_hybrid=false){
	srand(10);
	int query_size = config.query_size;
	int update_size = config.update_size;
	dynamic_workload.resize(query_size+update_size);
	dynamic_workload.assign(query_size+update_size,DQUERY);
	dynamic_workload[0]=DUPDATE;
	for(int i=1; i<update_size; ){
		int n = rand()%(query_size+update_size);
		if(dynamic_workload[n]!=DUPDATE){
			i++;
			dynamic_workload[n]=DUPDATE;
		}
	}
	
	if(if_hybrid){
		int ss_size=query_size/3;
		int topk_size=(query_size-ss_size)/2;
		int onehop_size=query_size-ss_size-topk_size;
		INFO(ss_size,topk_size,onehop_size);
		for(int i=0; i<ss_size; ){
			int n = rand()%(query_size+update_size);
			if(dynamic_workload[n]==DQUERY){
				i++;
				dynamic_workload[n]=DSSQUERY;
			}
		}
		for(int i=0; i<topk_size; ){
			int n = rand()%(query_size+update_size);
			if(dynamic_workload[n]==DQUERY){
				i++;
				dynamic_workload[n]=DTOPKQUERY;
			}
		}
		for(int i=0; i<query_size+update_size; i++){
			if(dynamic_workload[i]==DQUERY){
				dynamic_workload[i]=DOHQUERY;
			}
		}
		for(int op : dynamic_workload){
			//cout << op << endl;
		}
	
	}
}

void generate_queue_workload(bool if_hybrid=false){
	srand(10);
    printf("Query rate: %f, Update rate: %f, time: %f\n", config.query_arrRate, config.update_arrRate, config.queue_time);
	int query_size = config.query_arrRate*config.queue_time;
    config.query_size=query_size;
	int update_size = config.update_arrRate*config.queue_time;
    config.update_size=update_size;
    printf("Query num: %d, Update num: %d\n", query_size, update_size);
	dynamic_workload.resize(query_size+update_size);
	//dynamic_workload.assign(query_size+update_size, DQUERY);
    int mini_type, maxi_type;
    double mini;
    double maxi;
    if(query_size>update_size){
        mini=1.0/(double)(config.query_arrRate);
        mini_type=DQUERY;
        maxi=1.0/(double)(config.update_arrRate);
        maxi_type=DUPDATE;
        if(config.update_arrRate==0){
            maxi=config.queue_time+1;
        }
        if(config.query_arrRate==0){
            mini=config.queue_time+1;
        }
    }
    else{
        mini=1.0/(double)(config.update_arrRate);
        mini_type=DUPDATE;
        maxi=1.0/(double)(config.query_arrRate);
        maxi_type=DQUERY;
        if(config.query_arrRate==0){
            maxi=config.queue_time+1;
        }
        if(config.update_arrRate==0){
            mini=config.queue_time+1;
        }
    }
    
    double current_time=0;
    int current_workload=0;
    double next_mini_time=mini;
    double next_maxi_time=maxi;
    int mini_num=0;
    int maxi_num=0;
    while(current_time<=config.queue_time){
        current_time+=mini;
        if(current_time>config.queue_time){
            break;
        }
        if(current_time>=next_maxi_time){
            dynamic_workload[current_workload]=maxi_type;
            current_workload++;
            next_maxi_time+=maxi;
            maxi_num++;
        }
        if(current_time>=next_mini_time){
            dynamic_workload[current_workload]=mini_type;
            current_workload++;
            next_mini_time+=mini;
            mini_num++;
        }
    }
    printf("Generate Workload Finish\n");
    if(query_size>update_size){
        printf("Q: %d, U: %d\n",mini_num, maxi_num);
    }
    else{
        printf("Q: %d, U: %d\n",maxi_num, mini_num);
    }
    /*
	dynamic_workload[0]=DUPDATE;
	for(int i=1; i<update_size; ){
		int n = rand()%(query_size+update_size);
		if(dynamic_workload[n]!=DUPDATE){
			i++;
			dynamic_workload[n]=DUPDATE;
		}
	}
	*/
}

void dynamic_ssquery(NewGraph& newGraph){
	vector<int> queries;
    load_ss_query(queries);
    INFO(queries.size());
    unsigned int query_size = queries.size();
    query_size = min( query_size, config.query_size );
    int used_counter=0;
    
	ppr.init_keys(newGraph.getNumOfVertices());
	vector<pair<int,int>> updates;
	load_update(updates);
	INFO(updates.size());
	
	bool IF_OUTPUT=false;
	int query_count=0;
	int update_count=0;
	int test_k=0;
    fora_setting(newGraph.getNumOfVertices(), newGraph.getNumOfEdges());
    display_setting();
    fwd_idx.first.initialize(newGraph.getNumOfVertices());
    fwd_idx.second.initialize(newGraph.getNumOfVertices());
    reverse_idx.first.initialize(newGraph.getNumOfVertices());
    reverse_idx.second.initialize(newGraph.getNumOfVertices());
    inacc_idx.reserve(newGraph.getNumOfVertices());
    inacc_idx.assign(newGraph.getNumOfVertices(), 1);
    double theta=config.theta;
    double response_time_total=0;
    double sequence_time_total=0;
    double query_processing_time_total=0;
    Agenda_class agenda_worker(newGraph, config.epsilon, 0);
    double sequence_start=omp_get_wtime();

    //---------------------------------------Parameter Part---------------------------
    int num_total_workers=1;
    inacc_idx_all.resize(num_total_workers);                           // resize the columns
    for (auto &row : inacc_idx_all) { row.resize(newGraph.getNumOfVertices()+1); row.assign(newGraph.getNumOfVertices()+1, 1); }

    parallel_UM_W_status.Worker_needs_workload=false;
    parallel_UM_W_status.Available_cores=num_total_workers;
    parallel_UM_W_status.Total_cores=num_total_workers;
    parallel_UM_W_status.Update_Manager_is_awake=false;
    parallel_UM_W_status.TaskFinish=false;

    dynamic_workload_tail=0;

    std::atomic<bool> start_flag(false);
    int numProducerThreads=0;
    int numUpdateManagerThreds=0;
    int numConsumers=num_total_workers;


    for(int i=0; i<config.query_size+config.update_size; i++){
        cout<<"----------------------------------"<<endl;
        cout<<"operation "<<i<<endl;
        Timer timer(0);
        if(dynamic_workload[i]==DUPDATE){
            int u,v;
            
            u=updates[update_count].first;
            v=updates[update_count].second;
            update_count++;
            //double errorlimit=config.epsilon*1.0/graph.n/(1-config.alpha)*config.alpha*graph.g[u].size();
            const VertexIdType &idx_start = newGraph.get_neighbor_list_start_pos(u);
            const VertexIdType &idx_end = newGraph.get_neighbor_list_start_pos(u + 1);
            const VertexIdType degree = idx_end - idx_start;
            double errorlimit=double(std::max(1, degree))/newGraph.getNumOfVertices();
            errorlimit = errorlimit * config.r_b_scale * config.epsilon * 2;
            if(errorlimit==0){
                continue;
            }

            {
                Timer timer(11);
                double reverse_start=omp_get_wtime();
                reverse_push(u, newGraph, errorlimit, 1);
                double reverse_end=omp_get_wtime();
                
                printf("reverse time: %.12f\n", reverse_end-reverse_start);
            }
            newGraph.update(u, v);
            
            {
                Timer timer(12);	
                //for(long j=0; j<reverse_idx.first.occur.m_num; j++){
                for(long j=0; j<newGraph.getNumOfVertices(); j++){
                    //long id = reverse_idx.first.occur[j];
                    long id = j;
                    double pmin;
                    if(reverse_idx.first.exist(id))
                        pmin=min((reverse_idx.first[id]+errorlimit)*(1-config.alpha)/config.alpha,1.0);
                    else
                        pmin=min((errorlimit)*(1-config.alpha)/config.alpha,1.0);
                    //double pmin=(reverse_idx.first[id]+errorlimit*epsrate)/config.alpha;
                    const VertexIdType &idx_start_u = newGraph.get_neighbor_list_start_pos(u);
                    const VertexIdType &idx_end_u = newGraph.get_neighbor_list_start_pos(u + 1);
                    const VertexIdType degree_u = idx_end - idx_start;
                    inacc_idx[id]*=(1-pmin/degree_u);
                    //inacc_idx[id]-=pmin/graph.g[u].size();
                }
            }
        }
        else if(dynamic_workload[i]==DQUERY){
            double query_start=omp_get_wtime();
            //fora_query_lazy_dynamic(queries[query_count++], newGraph, theta);
            agenda_worker.agenda_query_lazy_dynamic_original_class(queries[query_count++], theta);
            double query_end=omp_get_wtime();
            query_processing_time_total+=query_end-query_start;
            response_time_total+=query_end-sequence_start;
            printf("query time: %.12f\n", query_end-query_start);
        }
    }
    sequence_time_total=omp_get_wtime()-sequence_start;
    // if(config.check_size>0){
    //     Timer::show();
    //     ofstream result_file;
    //     result_file.open("result/"+config.graph_alias+"/lazyup.txt");
    //     result_file<<config.check_size<<endl;

    //     if(config.check_from!=0){
    //         INFO(config.check_from);
    //         query_count=config.check_from;
    //     }
    //     for(int i=0; i<config.check_size; i++){
    //         cerr<<i;
    //         fora_query_lazy_dynamic(queries[query_count++], graph, theta, newGraph);
    //         output_imap(ppr, result_file, test_k);
    //     }
    //     cout<<endl;
    // }
    ofstream query_time_out_file;
    string s="AgendaOriginal";
    s+="_QueryRate_";
    s+=to_string(config.query_arrRate).substr(0, 5);
    s+="_UpdateRate_";
    s+=to_string(config.update_arrRate).substr(0, 5);
    s+="_ParaTime_";
    s+=to_string((int)(config.queue_time));
    s+="0322.txt";
    query_time_out_file.open(s);
    if (!query_time_out_file.is_open()) // check the file 
        cout << "cannot open file." << endl;
    query_time_out_file<<sequence_time_total<<endl;
    query_time_out_file<<response_time_total<<endl;
    query_time_out_file<<query_processing_time_total<<endl;
    cout<<"trw:"<<num_total_rw<<endl;
	cout<<"hrw:"<<num_hit_idx<<endl;
}
//-----------------------------------------------------------------------------------------------
void dynamic_test_victim_num(NewGraph& newGraph){
	vector<int> queries;
    load_ss_query(queries);
    INFO(queries.size());
    unsigned int query_size = queries.size();
    query_size = min( query_size, config.query_size );
    int used_counter=0;
    
	ppr.init_keys(newGraph.getNumOfVertices());
	vector<pair<int,int>> updates;
	load_update(updates);
	INFO(updates.size());
	
	bool IF_OUTPUT=false;
	int query_count=0;
	int update_count=100;
	int test_k=0;
    fora_setting(newGraph.getNumOfVertices(), newGraph.getNumOfEdges());
    display_setting();
    fwd_idx.first.initialize(newGraph.getNumOfVertices());
    fwd_idx.second.initialize(newGraph.getNumOfVertices());
    reverse_idx.first.initialize(newGraph.getNumOfVertices());
    reverse_idx.second.initialize(newGraph.getNumOfVertices());
    inacc_idx.reserve(newGraph.getNumOfVertices());
    inacc_idx.assign(newGraph.getNumOfVertices(), 0);
    double theta=config.theta;
    double response_time_total=0;
    double sequence_time_total=0;
    double query_processing_time_total=0;
    Agenda_class agenda_worker(newGraph, config.epsilon, 0);
    double sequence_start=omp_get_wtime();
    int check_round=1;
    vector<int> victim_num;
    victim_num.resize(config.update_size+10);
    victim_num.assign(config.update_size+10, 0);
    printf("check log(2/pf): %.12f\n", log(2/config.pfail));//*newGraph.getNumOfVertices()
    printf("check inaccuracy_bound: %.12f\n", 25*(1-theta)*config.beta_PFS*((2+2*config.epsilon)*log(2/config.pfail)/config.epsilon/config.theta/config.theta));//*newGraph.getNumOfVertices()
    printf("check rbmax: %.12f\n", config.r_b_scale * config.epsilon * 2/newGraph.getNumOfVertices());
    printf("check rmax: %.12f\n", config.rmax);
    for(int k=0; k<check_round; k++){
        for(int i=0; i<config.update_size; i++){
            cout<<"----------------------------------"<<endl;
            cout<<"round "<<k<<" operation "<<i<<endl;
            Timer timer(0);
            double error_Shuffling_bound=(config.epsilon*(1-theta)/newGraph.getNumOfVertices())*config.beta_PFS;
            double inaccuracy_bound=((config.epsilon*(1-theta)/newGraph.getNumOfVertices())*config.beta_PFS)/config.rmax/newGraph.getNumOfVertices();
            printf("check inaccuracy_bound: %.12f\n", inaccuracy_bound);
            if(dynamic_workload[i]==DUPDATE){
                int u,v;
                
                u=updates[update_count].first;
                v=updates[update_count].second;
                update_count++;
                //double errorlimit=config.epsilon*1.0/graph.n/(1-config.alpha)*config.alpha*graph.g[u].size();
                const VertexIdType &idx_start = newGraph.get_neighbor_list_start_pos(u);
                const VertexIdType &idx_end = newGraph.get_neighbor_list_start_pos(u + 1);
                const VertexIdType degree = idx_end - idx_start;
                double errorlimit=double(std::max(1, degree))/newGraph.getNumOfVertices();
                errorlimit = errorlimit * config.r_b_scale * config.epsilon * 2;
                printf("check rbmax-inaccuracy: %.12f\n", errorlimit/(config.alpha*degree));
                if(errorlimit==0){
                    continue;
                }

                {
                    Timer timer(11);
                    double reverse_start=omp_get_wtime();
                    reverse_push(u, newGraph, errorlimit, 1);
                    double reverse_end=omp_get_wtime();
                    
                    printf("reverse time: %.12f\n", reverse_end-reverse_start);
                }
                newGraph.update(u, v);
                int check_temp=0;
                int check_temp_1=0;
                int check_temp_2=0;
                int check_temp_3=0;
                int check_degree=0;
                double check_reverse_idx=0;
                {
                    Timer timer(12);	
                    //for(long j=0; j<reverse_idx.first.occur.m_num; j++){
                    for(long j=0; j<newGraph.getNumOfVertices(); j++){
                        //long id = reverse_idx.first.occur[j];
                        long id = j;
                        double pmin;
                        if(reverse_idx.first.exist(id)){
                            pmin=min((reverse_idx.first[id]+errorlimit)/(config.alpha),1.0);//*(1-config.alpha)
                            check_temp_3++;
                            if(reverse_idx.first[id]/(config.alpha*degree)<=inaccuracy_bound){
                                check_temp_1++;
                                check_reverse_idx+=reverse_idx.first[id]/(config.alpha*degree);
                            }
                        }
                        else{
                            pmin=errorlimit/(config.alpha);//errorlimit*(1-config.alpha)/config.alpha
                            check_temp_2++;
                            if(newGraph.get_neighbor_list_start_pos(j+1)-newGraph.get_neighbor_list_start_pos(j)<=7){
                                check_degree++;
                            }
                        }
                        
                        //double pmin=(reverse_idx.first[id]+errorlimit*epsrate)/config.alpha;
                        const VertexIdType &idx_start_u = newGraph.get_neighbor_list_start_pos(u);
                        const VertexIdType &idx_end_u = newGraph.get_neighbor_list_start_pos(u + 1);
                        const VertexIdType degree_u = idx_end - idx_start;
                        inacc_idx[id]+=(pmin/degree_u);
                        if(pmin/degree_u>inaccuracy_bound/(newGraph.get_neighbor_list_start_pos(j+1)-newGraph.get_neighbor_list_start_pos(j))){
                            check_temp++;
                        }
                        //inacc_idx[id]-=pmin/graph.g[u].size();
                    }
                    check_reverse_idx/=check_temp_1;
                    printf("check have reverse, num: %d\n", check_temp_3);
                    printf("check degree<=7, num: %d\n", check_degree);
                    printf("check only reverse, num: %d, aver: %.12f \n", check_temp_1+check_temp_2, check_reverse_idx);
                    printf("check victim num is: %d\n", check_temp);
                }
            }
            
            for(long j=0; j<newGraph.getNumOfVertices(); j++){
                const VertexIdType &idx_start_next = newGraph.get_neighbor_list_start_pos(j);
                const VertexIdType &idx_end_next = newGraph.get_neighbor_list_start_pos(j + 1);
                int degree_next = idx_end_next - idx_start_next;
                if(degree_next==0) 
                    degree_next=1;
                if((inacc_idx[j])*degree_next>(inaccuracy_bound)){ ///
                    victim_num[i]++;
                }
            }
        }
        for(long j=0; j<newGraph.getNumOfVertices(); j++){
            inacc_idx[j]=0;
            //inacc_idx[id]-=pmin/graph.g[u].size();
        }

    }
    for(int i=0; i<config.update_size; i++){
        printf("U %d, check victim num: %d\n", i, victim_num[i]);
        victim_num[i]=victim_num[i]/check_round;
    }
    sequence_time_total=omp_get_wtime()-sequence_start;
    // if(config.check_size>0){
    //     Timer::show();
    //     ofstream result_file;
    //     result_file.open("result/"+config.graph_alias+"/lazyup.txt");
    //     result_file<<config.check_size<<endl;

    //     if(config.check_from!=0){
    //         INFO(config.check_from);
    //         query_count=config.check_from;
    //     }
    //     for(int i=0; i<config.check_size; i++){
    //         cerr<<i;
    //         fora_query_lazy_dynamic(queries[query_count++], graph, theta, newGraph);
    //         output_imap(ppr, result_file, test_k);
    //     }
    //     cout<<endl;
    // }
    ofstream query_time_out_file;
    string s="PFSTestVictimNum";
    s+="_QueryRate_";
    s+=to_string(config.query_arrRate).substr(0, 5);
    s+="_UpdateRate_";
    s+=to_string(config.update_arrRate).substr(0, 5);
    s+="_ParaTime_";
    s+=to_string((int)(config.queue_time));
    s+="_Graph_";
    s+=config.graph_alias;
    s+="_0412.txt";
    query_time_out_file.open(s);
    if (!query_time_out_file.is_open()) // check the file 
        cout << "cannot open file." << endl;
    for(int i=0; i<config.update_size; i++){
        query_time_out_file<<victim_num[i]<<endl;
    }
    cout<<"trw:"<<num_total_rw<<endl;
	cout<<"hrw:"<<num_hit_idx<<endl;
}


//-------------------------Parallel Part--------------------------------------------
void ConsumeItem_original(NewGraph& newGraph, const DY_worktask &single_task, double start_time, Agenda_class &agenda_worker, int total_worker_number,  double &response_time_total, double &query_processing_time_total){
    
    // printf("workload size: %d\n",parallel_dynamic_workload.workload.size());
    double OMP_check_query_time = omp_get_wtime();//OMP_TIME_START
    if(single_task.type==DUPDATE) {
        double OMP_check_reverse_start=omp_get_wtime();
        //write mutex
        int u,v;
        int update_number_index;
        //printf("|||W: Update|||\n");
        u=single_task.update_start;
        v=single_task.update_end;
        update_number_index=single_task.update_number;
        const VertexIdType &idx_start = newGraph.get_neighbor_list_start_pos(u);
        const VertexIdType &idx_end = newGraph.get_neighbor_list_start_pos(u + 1);
        const VertexIdType degree = idx_end - idx_start;
        double errorlimit=double(std::max(1, degree))/newGraph.getNumOfVertices();
        errorlimit = errorlimit * config.r_b_scale * config.epsilon * 2;

        reverse_push(u, newGraph, errorlimit, 1);
        newGraph.update(u, v);
        {
            for(long j=0; j<newGraph.getNumOfVertices(); j++){
                //long id = reverse_idx.first.occur[j];
                long id = j;
                double pmin;
                if(reverse_idx.first.exist(id))
                    pmin=min((reverse_idx.first[id]+errorlimit)*(1-config.alpha)/config.alpha,1.0);
                else
                    pmin=min((errorlimit)*(1-config.alpha)/config.alpha,1.0);
                //double pmin=(reverse_idx.first[id]+errorlimit*epsrate)/config.alpha;
                const VertexIdType &idx_start_u = newGraph.get_neighbor_list_start_pos(u);
                const VertexIdType &idx_end_u = newGraph.get_neighbor_list_start_pos(u + 1);
                const VertexIdType degree_u = idx_end - idx_start;
                //inacc_idx[id]*=(1-pmin/degree_u);
                //inacc_idx[id]-=pmin/graph.g[u].size();
                for(int i=0; i<total_worker_number; i++){
                    inacc_idx_all[i][id]*=(1-pmin/degree_u);
                }
            }
        }
        //cout<< "UPDATE Finish"<<endl;	
        double OMP_check_reverse_end=omp_get_wtime();
        printf("Update finish\n", OMP_check_reverse_end-OMP_check_reverse_start);
    } 
    else if (single_task.type==DQUERY) {
        double OMP_check_Query_start=omp_get_wtime();
        double theta=config.theta;
        agenda_worker.agenda_query_lazy_dynamic_original_class(single_task.source, theta);
        double OMP_check_Query_end=omp_get_wtime();
        response_time_total+=OMP_check_Query_end-start_time;
        query_processing_time_total+=OMP_check_Query_end-OMP_check_Query_start;
        printf("Query finish\n", OMP_check_Query_end-OMP_check_Query_start);
    }
    double OMP_check_query_time_end = omp_get_wtime();
    // printf("Thread %d :\n", omp_get_thread_num()); 
    
}

void dynamic_ssquery_parallel(NewGraph& newGraph, int num_total_workers){
	vector<int> queries;
    load_ss_query(queries);
    INFO(queries.size());
    unsigned int query_size = queries.size();
    query_size = min( query_size, config.query_size );
    int used_counter=0;
    
	ppr.init_keys(newGraph.getNumOfVertices());
	vector<pair<int,int>> updates;
	load_update(updates);
	INFO(updates.size());
	
	bool IF_OUTPUT=false;
	int query_count=0;
	int update_count=0;
	int test_k=0;
    fora_setting(newGraph.getNumOfVertices(), newGraph.getNumOfEdges());
    display_setting();
    fwd_idx.first.initialize(newGraph.getNumOfVertices());
    fwd_idx.second.initialize(newGraph.getNumOfVertices());
    reverse_idx.first.initialize(newGraph.getNumOfVertices());
    reverse_idx.second.initialize(newGraph.getNumOfVertices());
    inacc_idx.reserve(newGraph.getNumOfVertices());
    inacc_idx.assign(newGraph.getNumOfVertices(), 1);
    double theta=config.theta;
    double response_time_total=0;
    double sequence_time_total=0;
    double query_processing_time_total=0;
    //Agenda_class agenda_worker(newGraph, config.epsilon, 0);
    double sequence_start=omp_get_wtime();
    std::vector<std::thread> threads;
    
//---------------------------------------Parameter Part---------------------------
    inacc_idx_all.resize(num_total_workers);                           // resize the columns
    for (auto &row : inacc_idx_all) { row.resize(newGraph.getNumOfVertices()+1); row.assign(newGraph.getNumOfVertices()+1, 1); }

    parallel_UM_W_status.Worker_needs_workload=false;
    parallel_UM_W_status.Available_cores=num_total_workers;
    parallel_UM_W_status.Total_cores=num_total_workers;
    parallel_UM_W_status.Update_Manager_is_awake=false;
    parallel_UM_W_status.TaskFinish=false;

    dynamic_workload_tail=0;

    std::atomic<bool> start_flag(false);
    int numProducerThreads=0;
    int numUpdateManagerThreds=0;
    int numConsumers=num_total_workers;


//---------------------------------------Mutex Part---------------------------
    std::mutex pop_queue_mtx;
    std::mutex write_mtx;
    std::mutex update_mtx;
    std::mutex read_mtx;
    int readCnt = 0;
    int popCnt = 0;

//---------------------------------------Worker Part--------------------------------
    config.total_task_size=config.query_size+config.update_size;
    for (uint64_t thread_i = numProducerThreads+numUpdateManagerThreds; thread_i < numConsumers+numUpdateManagerThreds+numProducerThreads; ++thread_i) {
        threads.push_back(std::thread([&, thread_i]() {
            while (!start_flag){
                sleep(10);
            }
            // thread_set.insert(std::this_thread::get_id());
            Agenda_class agenda_worker(newGraph, config.epsilon, (thread_i-numProducerThreads-numUpdateManagerThreds));
            int temp_pop_cnt;
            //DY_worktask single_task;
            DY_worktask single_task;
            bool check_pop_single_task=false;
            while(true) {
                pop_queue_mtx.lock();

                if (popCnt>=config.total_task_size||parallel_UM_W_status.TaskFinish==true) {
                    parallel_UM_W_status.TaskFinish=true;
                    pop_queue_mtx.unlock();
                    //std::cout<<"finish!"<<endl;
                    break;
                }
                
                int workload_type=dynamic_workload[dynamic_workload_tail];
                dynamic_workload_tail++;
                
                parallel_UM_W_status.lck.lock();
                parallel_UM_W_status.Available_cores-=1;
                printf("Th %d start\n", thread_i-numProducerThreads-numUpdateManagerThreds);
                parallel_UM_W_status.lck.unlock();
                //std::cout << "Worker" << i<<": "<< std::this_thread::get_id()<< " get " << temp_pop_cnt << "^th item doing " << task_type <<"." << std::endl;
                
                temp_pop_cnt = popCnt;
                popCnt++;
                //printf("Th %d 1\n", thread_i-numProducerThreads-numUpdateManagerThreds);
                //pop_queue_mtx.unlock();

                if(workload_type == DQUERY){
                    single_task.type=DQUERY;
                    query_count++;
                    single_task.source=queries[query_count];
                    printf("Th %d Q1.1\n", thread_i-numProducerThreads-numUpdateManagerThreds);
                    read_mtx.lock();
                    if (++readCnt==1){
                        //write_mtx.lock();
                    }
                    read_mtx.unlock();
                    pop_queue_mtx.unlock();
                } 
                else if(workload_type == DUPDATE) {
                    single_task.type=DUPDATE;
                    update_count++;
                    single_task.update_start=updates[update_count].first;
                    single_task.update_end=updates[update_count].second;
                    
                    //pop_queue_mtx.lock();
                    printf("Th %d U1.2\n", thread_i-numProducerThreads-numUpdateManagerThreds);
                    for(int i=0; i<1; ){
                        if(readCnt==0){
                            break;
                        }
                        usleep(10);
                    }
                    write_mtx.lock();
                    printf("Th %d U1.3\n", thread_i-numProducerThreads-numUpdateManagerThreds);
                }
                printf("Th %d 2\n", thread_i-numProducerThreads-numUpdateManagerThreds);
                //std::cout << "Consumer thread " << i<<": "<< std::this_thread::get_id()<< " is consuming the " << temp_pop_cnt << "^th item doing " << task_type <<"." << std::endl;
                
                ConsumeItem_original(newGraph,  single_task, sequence_start, agenda_worker, numConsumers, response_time_total, query_processing_time_total);

                printf("Th %d 3\n", thread_i-numProducerThreads-numUpdateManagerThreds);
                parallel_UM_W_status.lck.lock();
                parallel_UM_W_status.Available_cores+=1;
                parallel_UM_W_status.lck.unlock();
                //printf("NumOps: %d PopCnt: %d\n", numOps, popCnt);
                printf("Th %d, check system status cores: %d, readCNT:%d\n",thread_i-numProducerThreads-numUpdateManagerThreds ,parallel_UM_W_status.Available_cores, readCnt);
                
                if(single_task.type == DQUERY){
                    read_mtx.lock();
                    if (--readCnt==0){
                        //write_mtx.unlock();
                    }
                    read_mtx.unlock();
                } else if(single_task.type == DUPDATE) {
                    pop_queue_mtx.unlock();
                    write_mtx.unlock();
                }
            }
            printf("Thread: %d, finish!\n", thread_i-numProducerThreads-numUpdateManagerThreds);
        }));
    }

    start_flag = true;
    for (auto &thread : threads) {
        thread.join();
    }
    
    sequence_time_total=omp_get_wtime()-sequence_start;
    //-------------------------------------------------------------------------------------------
    ofstream query_time_out_file;
    string folder_path="./";
    folder_path+=config.graph_alias;
    folder_path+="/";
    boost::filesystem::create_directories(folder_path);    
    string s="AgendaParallel";
    s+="_QueryRate_";
    s+=to_string(config.query_arrRate).substr(0, 5);
    s+="_UpdateRate_";
    s+=to_string(config.update_arrRate).substr(0, 5);
    s+="_ParaTime_";
    s+=to_string((int)(config.queue_time));
    s+="_WorkerNum_";
    s+=to_string((int)(config.total_worker_num));
    s+="0324.txt";
    string final_path_txt=folder_path+s;
    query_time_out_file.open(final_path_txt);
    if (!query_time_out_file.is_open()) // check the file 
        cout << "cannot open file." << endl;
    query_time_out_file<<sequence_time_total<<endl;
    query_time_out_file<<response_time_total<<endl;
    query_time_out_file<<query_processing_time_total<<endl;
    cout<<"trw:"<<num_total_rw<<endl;
	cout<<"hrw:"<<num_hit_idx<<endl;
}
//-----------------------------------------------------Error Push Part--------------------------------------------------------------
//-----------------------------------------------------Error Push Part--------------------------------------------------------------
//-----------------------------------------------------Error Push Part--------------------------------------------------------------
//-----------------------------------------------------Error Push Part--------------------------------------------------------------
//-----------------------------------------------------Error Push Part--------------------------------------------------------------

void ConsumeItem_ErrorPush(NewGraph& newGraph, const DY_worktask &single_task, double start_time, Agenda_class &agenda_worker, int total_worker_number, double &response_time_total, double &query_processing_time_total){
    
    // printf("workload size: %d\n",parallel_dynamic_workload.workload.size());
    double OMP_check_query_time = omp_get_wtime();//OMP_TIME_START
    if(single_task.type==DUPDATE) {
        double OMP_check_reverse_start=omp_get_wtime();
        //write mutex
        int u,v;
        int update_number_index;
        //printf("|||W: Update|||\n");
        u=single_task.update_start;
        v=single_task.update_end;
        update_number_index=single_task.update_number;
        const VertexIdType &idx_start = newGraph.get_neighbor_list_start_pos(u);
        const VertexIdType &idx_end = newGraph.get_neighbor_list_start_pos(u + 1);
        const VertexIdType degree = idx_end - idx_start;
        double errorlimit=double(std::max(1, degree))/newGraph.getNumOfVertices();
        errorlimit = errorlimit * config.r_b_scale * config.epsilon * 2;

        //reverse_push(u, newGraph, errorlimit, 1);
        newGraph.update_out_neighbor(u, v);
        // {
        //     for(long j=0; j<newGraph.getNumOfVertices(); j++){
        //         //long id = reverse_idx.first.occur[j];
        //         long id = j;
        //         double pmin;
        //         if(reverse_idx.first.exist(id))
        //             pmin=min((reverse_idx.first[id]+errorlimit)*(1-config.alpha)/config.alpha,1.0);
        //         else
        //             pmin=min((errorlimit)*(1-config.alpha)/config.alpha,1.0);
        //         //double pmin=(reverse_idx.first[id]+errorlimit*epsrate)/config.alpha;
        //         const VertexIdType &idx_start_u = newGraph.get_neighbor_list_start_pos(u);
        //         const VertexIdType &idx_end_u = newGraph.get_neighbor_list_start_pos(u + 1);
        //         const VertexIdType degree_u = idx_end - idx_start;
        //         //inacc_idx[id]*=(1-pmin/degree_u);
        //         //inacc_idx[id]-=pmin/graph.g[u].size();
        //         for(int i=0; i<total_worker_number; i++){
        //             inacc_idx_all[i][id]*=(1-pmin/degree_u);
        //         }
        //     }
        // }
        //cout<< "UPDATE Finish"<<endl;	
        //--------------------------update inacc--------------------------------------------
        if(single_task.idx_number==0){
            int worker_inacc_tail_temp=inacc_idx_map_worker.tail;
            //inacc_idx_map_worker.inacc[]
            for(int id_temp=0; id_temp<newGraph.getNumOfVertices(); id_temp++){
                long id=id_temp;
                for(int k=0; k<total_worker_number; k++){
                    inacc_idx_all[k][id]*=inacc_idx_map_worker.inacc[worker_inacc_tail_temp][id];
                }
            }
        }
        //----------------------------------------------------------------------------------
        double OMP_check_reverse_end=omp_get_wtime();
        printf("Update finish\n", OMP_check_reverse_end-OMP_check_reverse_start);
    } 
    else if (single_task.type==DQUERY) {
        double OMP_check_Query_start=omp_get_wtime();
        double theta=config.theta;
        if(single_task.move_BehindOrAfter==1){
            printf("Original Agenda\n");
            double OMP_check_query_start=omp_get_wtime();
            agenda_worker.agenda_query_lazy_dynamic_original_class(single_task.source, theta);
            double OMP_check_query_end=omp_get_wtime();
            response_time_total+=OMP_check_query_end-start_time;
            query_processing_time_total+=OMP_check_query_end-OMP_check_query_start;
            //printf("Query Finish: %.12f\n", OMP_check_query_start-OMP_check_query_end);
        }
        else if(single_task.move_BehindOrAfter==0){
            printf("Error Push Agenda\n");
            double OMP_check_query_start=omp_get_wtime();
            agenda_worker.agenda_query_lazy_dynamic_error_push_class(single_task.source, theta, single_task.update_number, single_task.idx_number);
            double OMP_check_query_end=omp_get_wtime();
            response_time_total+=OMP_check_query_end-start_time;
            query_processing_time_total+=OMP_check_query_end-OMP_check_query_start;
        }
        double OMP_check_Query_end=omp_get_wtime();
        printf("Query finish\n", OMP_check_Query_end-OMP_check_Query_start);
    }
    double OMP_check_query_time_end = omp_get_wtime();
    // printf("Thread %d :\n", omp_get_thread_num()); 
    
}

void UpdateManager_reverse_push(ReversePushidx &reverse_idx_push, int t, NewGraph &graph, double myeps, double init_residual = 1) {
    reverse_idx_push.first.clean();
    reverse_idx_push.second.clean();

    static unordered_map<int, bool> idx;

    vector<int> q;
    q.reserve(graph.getNumOfVertices());
    q.push_back(-1);
    unsigned long left = 1;

    //double myeps = config.rmax;

    q.push_back(t);
    reverse_idx_push.second.insert(t, init_residual);
    //printf("start real reverse push\n");
    idx[t] = true;
    while (left < q.size()) {
        //printf("check v: %d\n", q[left]);
        int v = q[left];
        idx[v] = false;
        left++;
        if (reverse_idx_push.second[v] < myeps)
            break;

        if(reverse_idx_push.first.notexist(v))
            reverse_idx_push.first.insert(v, reverse_idx_push.second[v]*config.alpha);
        else
            reverse_idx_push.first[v] += reverse_idx_push.second[v]*config.alpha;

        double residual = (1 - config.alpha) * reverse_idx_push.second[v];
        reverse_idx_push.second[v] = 0;
        const VertexIdType &idx_start = graph.get_in_neighbor_list_start_pos(v);
        const VertexIdType &idx_end = graph.get_in_neighbor_list_start_pos(v + 1);
        const VertexIdType degree = idx_end - idx_start;
        //printf("deal next: %d\n", degree);
        if(degree>0){
            for (int j = idx_start; j < idx_end; ++j) {
                VertexIdType next = graph.getInNeighbor_PFS(j);
                //printf("check next: %d\n", next);
                // const VertexIdType &idx_start_next = graph.get_neighbor_list_start_pos(next);
                // const VertexIdType &idx_end_next = graph.get_neighbor_list_start_pos(next + 1);
                VertexIdType degree_next = graph.get_out_degree_shuffling_controller(next);
                //printf("check next: %d, degree: %d\n", next, degree_next);
                if(reverse_idx_push.second.notexist(next))
                    reverse_idx_push.second.insert(next, residual/degree_next);
                else
                    reverse_idx_push.second[next] += residual/degree_next;

                if (reverse_idx_push.second[next] > myeps && idx[next] != true) {
                    // put next into q if next is not in q
                    idx[next] = true;//(int) q.size();
                    q.emplace_back(next);
                }
            }
        }
    }
}

void UpdateManager_Update_Inacc(map<int, vector<double>>& inacc_idx_map, NewGraph& graph_UM, int inacc_map_head, double Rbmax, int u){
    map<int, vector<double>>::iterator it=inacc_idx_map.end();
    //printf("inacc_size: %d, updateCount: %d\n",(int)(it->first),  updateCount);
    if((int)(it->first)<inacc_map_head){
        inacc_idx_map[inacc_map_head].reserve(graph_UM.getNumOfVertices());
    }
    {
        for(long j=0; j<graph_UM.getNumOfVertices(); j++){
            long id=j;
            double pmin;
            if(reverse_idx_UM.first.exist(id)){
                pmin=min((reverse_idx_UM.first[id]+Rbmax)*(1-config.alpha)/config.alpha,1.0);
            }
            else{
                pmin=min((Rbmax)*(1-config.alpha)/config.alpha,1.0);
            }
            const VertexIdType &idx_start_u = graph_UM.get_neighbor_list_start_pos(u);
            const VertexIdType &idx_end_u = graph_UM.get_neighbor_list_start_pos(u + 1);
            const VertexIdType degree_u = idx_end_u - idx_start_u;
            inacc_idx_map[inacc_map_head][id]*=(1-pmin/degree_u);
        }
        //printf("Update inacc finish\n");
    }
}

void ShufflingController_batch_order_queue(NewGraph& graph_UM, DY_worktask &single_task, list<DY_worktask> &orderedList_update, list<DY_worktask> &orderedList_query, list<DY_worktask> &orderedList_storage, int &query_number_in_round, int &update_number_in_round, int &inacc_map_head, int &popCnt_UM, double theta, int num_total_worker){
    if(single_task.type==DQUERY){
        //std::cout<<"---------------------------------------------------------------"<<endl;
        //std::cout<<"UM Query"<<endl;
        long id;
        bool flag;
        int update_jumped_number=0;
        double error_sum=0;
        double error_bound=config.epsilon/graph_UM.getNumOfVertices()*(1-theta)*config.beta_PFS;
        int query_jumped_number=0;
        //query_number_in_round++;
        single_task.query_number=query_number_in_round;
        single_task.update_number=update_number_in_round;
        popCnt_UM++;
        //list<DY_worktask>::iterator orderedList_it;
        // if no update before, directly insert query into ordered list
        if(orderedList_query.size()==0){
            orderedList_query.push_back(single_task);
            single_task.move_BehindOrAfter=1;
        }
        else{
            // else, check the inacc 
            id=single_task.source;
            //orderedList_it=orderedList_query.end();
            flag=false;
            error_sum=0;
            error_sum += (1-inacc_idx_map_UM[inacc_map_head][id]);
            printf("\t\t\t\t\tcheck error sum: %.12f\n", error_sum);
            //printf("UM manage Source: %d, error_sum: %.12f\n",single_task.source, error_sum);
            if(update_number_in_round<=0){ //if error_sum less than error bound, move forward
                {
                    //printf("check error_sum: %.16f\ncheck error_bound: %.16f\n", error_sum, error_bound);
                    single_task.error_after=error_sum;
                    single_task.move_BehindOrAfter=1; //move forward
                }
                orderedList_query.push_back(single_task);
            }
            else{ //if error_sum bigger than error bound, error-push
                single_task.idx_number=inacc_map_head;
                inacc_map_head++;
                single_task.error_after=0;
                single_task.move_BehindOrAfter=0; //Error Push
                orderedList_query.push_back(single_task);
                inacc_idx_map_UM[inacc_map_head].assign(inacc_idx_map_UM[inacc_map_head-1].begin(), inacc_idx_map_UM[inacc_map_head-1].end());
            }
        }
    }
    else if(single_task.type==DUPDATE){
        //std::cout<<"It's Update"<<endl;
        int u,v;
        //update_number_in_round++;
        single_task.query_number=query_number_in_round;
        single_task.update_number=update_number_in_round;
        popCnt_UM++;
        u=single_task.update_start;
        v=single_task.update_end;
        const VertexIdType &idx_start_u = graph_UM.get_neighbor_list_start_pos(u);
        const VertexIdType &idx_end_u = graph_UM.get_neighbor_list_start_pos(u + 1);
        const VertexIdType degree = idx_end_u - idx_start_u;
        double errorlimit=double(std::max(1, degree))/graph_UM.getNumOfVertices();
        errorlimit = errorlimit * config.r_b_scale * config.epsilon * 2; //r_b max
        //double optimize_factor=config.epsilon*0.5;
        double optimize_factor=1;
        double error_bound=config.epsilon/graph_UM.getNumOfVertices()*(1-theta);
        printf("\t\t\t\t\tShuffling Controller start Update\n");
        //printf("\t\t\t\t\t\t\trbmax: %.12f, error_bound: %.12f\n", errorlimit*epsrate*optimize_factor, error_bound);
        {
            double UM_check_reverse_start=omp_get_wtime();
            UpdateManager_reverse_push(reverse_idx_UM, u, graph_UM, errorlimit);
            double UM_check_reverse_end=omp_get_wtime();
            printf("\t\t\t\t\t\t\tUM_Batch-CTIME check UM reverse: %.6f\n", (UM_check_reverse_end-UM_check_reverse_start));
        }
        //std::cout<<"Reverse Push end-UM"<<endl;
        int count=0;
        
        //-----------------------update inacc-------------------------------------
        UpdateManager_Update_Inacc(inacc_idx_map_UM, graph_UM, inacc_map_head, errorlimit, u);

        //-----------------------update graph-------------------------------------
        graph_UM.update_in_neighbor(u, v);
        orderedList_update.push_back(single_task);
        //printf("\t\t\t\t\t\t\tPush an update\n");
    }
}

void dynamic_ssquery_parallel_shuffling(NewGraph& newGraph, int num_total_workers, int num_total_shuffling_controller){
	vector<int> queries;
    load_ss_query(queries);
    INFO(queries.size());
    unsigned int query_size = queries.size();
    query_size = min( query_size, config.query_size );
    int used_counter=0;
    
	ppr.init_keys(newGraph.getNumOfVertices());
	vector<pair<int,int>> updates;
	load_update(updates);
	INFO(updates.size());
	
	bool IF_OUTPUT=false;
	int query_count=0;
	int update_count=0;
	int test_k=0;
    fora_setting(newGraph.getNumOfVertices(), newGraph.getNumOfEdges());
    display_setting();
    fwd_idx.first.initialize(newGraph.getNumOfVertices());
    fwd_idx.second.initialize(newGraph.getNumOfVertices());
    reverse_idx.first.initialize(newGraph.getNumOfVertices());
    reverse_idx.second.initialize(newGraph.getNumOfVertices());
    inacc_idx.reserve(newGraph.getNumOfVertices());
    inacc_idx.assign(newGraph.getNumOfVertices(), 1);
    double theta=config.theta;
    double response_time_total=0;
    double sequence_time_total=0;
    double query_processing_time_total=0;
    //Agenda_class agenda_worker(newGraph, config.epsilon, 0);
    double sequence_start=omp_get_wtime();
    std::vector<std::thread> threads;
    printf("Main part start\n");
//---------------------------------------Parameter Part---------------------------
    inacc_idx_all.resize(num_total_workers);           // resize the columns
    for (auto &row : inacc_idx_all) { row.resize(newGraph.getNumOfVertices()+1); row.assign(newGraph.getNumOfVertices()+1, 1); }

    parallel_UM_W_status.Worker_needs_workload=false;
    parallel_UM_W_status.Available_cores=num_total_workers;
    parallel_UM_W_status.Total_cores=num_total_workers;
    parallel_UM_W_status.Update_Manager_is_awake=false;
    parallel_UM_W_status.TaskFinish=false;

    dynamic_workload_tail=0;

    std::atomic<bool> start_flag(false);
    int numProducerThreads=0;
    int numUpdateManagerThreds=num_total_shuffling_controller;
    int numConsumers=num_total_workers;

    for(int i=0; i<num_total_workers+5; i++){
        inacc_idx_map_UM[i].reserve(newGraph.getNumOfVertices());
        inacc_idx_map_UM[i].resize(newGraph.getNumOfVertices());
        inacc_idx_map_UM[i].assign(newGraph.getNumOfVertices(), 1);
        inacc_idx_map_worker.inacc[i].reserve(newGraph.getNumOfVertices());
        inacc_idx_map_worker.inacc[i].resize(newGraph.getNumOfVertices());
        inacc_idx_map_worker.inacc[i].assign(newGraph.getNumOfVertices(), 1);
    }
    config.total_task_size=config.query_size+config.update_size;

    UpdateManager_batch_recorder.update_start.resize(newGraph.getNumOfVertices()/2);
    UpdateManager_batch_recorder.update_start.assign(newGraph.getNumOfVertices()/2, -1);
    UpdateManager_batch_recorder.update_end.resize(newGraph.getNumOfVertices()/2);
    UpdateManager_batch_recorder.update_end.assign(newGraph.getNumOfVertices()/2, -1);
    UpdateManager_batch_recorder.head=0;
    UpdateManager_batch_recorder.tail=0;

    Workers_batch_recorder.node_type.resize(newGraph.getNumOfVertices()+1);
    Workers_batch_recorder.node_type.assign(newGraph.getNumOfVertices()+1, 0);
    Workers_batch_recorder.update_start.resize(newGraph.getNumOfVertices()/2);
    Workers_batch_recorder.update_start.assign(newGraph.getNumOfVertices()/2, -1);
    Workers_batch_recorder.update_end.resize(newGraph.getNumOfVertices()/2);
    Workers_batch_recorder.update_end.assign(newGraph.getNumOfVertices()/2, -1);
    Workers_batch_recorder.head=0;
    Workers_batch_recorder.tail=0;

    reverse_idx_UM.first.initialize(newGraph.getNumOfVertices());
    reverse_idx_UM.second.initialize(newGraph.getNumOfVertices());
//---------------------------------------Mutex Part---------------------------
    std::mutex pop_queue_mtx;
    std::mutex write_mtx;
    std::mutex update_mtx;
    std::mutex read_mtx;
    int readCnt = 0;
    int popCnt = 0;
    int popCnt_UM=0;
    printf("multiple thread part start now\n");
//---------------------------------------Shuffling Controller Part--------------------------------
    for (int thread_i=numProducerThreads; thread_i<numProducerThreads+num_total_shuffling_controller; thread_i++) {
        threads.push_back(std::thread([&, thread_i]() {
            while (!start_flag){
                sleep(10);
            }
            //std::cout<<"UpdateManager has"<<numOps<<" work to deal"<<endl;
            printf("\t\t\tShuffling Controller start \n");
            int popCnt=0;
            int inacc_map_head=0;
            DY_worktask single_task;
            DY_worktask single_task_for_push;
            bool check_pop_single_task;
            
            double errorlimit;
            double theta=0.8;
            double epsrate=0.5;
            list<DY_worktask> orderedList_update;
            list<DY_worktask> orderedList_query;
            list<DY_worktask> orderedList_storage;
            int totalOrderedListSize=0;
            int totalQueryListSize=0;
            int totalStorageListSize=0;
            
            //int min_task_num=2;
            //int upper_bound_orderedList_size=40;
            int query_number_in_round=0;
            int update_number_in_round=0;
            int update_u, update_v;

            // while(true){
            //     usleep(100);
            //     if(parallel_UM_W_status.Update_Manager_is_awake==true){
            //         break;
            //     }
            //     if(parallel_UM_W_status.TaskFinish==true){
            //         break;
            //     }
            // }
            //DY_worktask single_task;
            while(true){
                if(popCnt_UM>=config.query_size+config.update_size){
                    parallel_UM_W_status.TaskFinish=true;
                }
                if(parallel_UM_W_status.TaskFinish==true){
                    break;
                }
                printf("\t\t\tHave not end \n");
                int workload_type=dynamic_workload[dynamic_workload_tail];
                dynamic_workload_tail++;
                if(workload_type == DQUERY){
                    printf("\t\t\tGot a Query \n");
                    single_task.type=DQUERY;
                    query_count++;
                    single_task.source=queries[query_count];
                    popCnt_UM++;
                }
                else if(workload_type == DUPDATE){
                    printf("\t\t\tGot an Update \n");
                    single_task.type=DUPDATE;
                    update_count++;
                    single_task.update_start=updates[update_count].first;
                    single_task.update_end=updates[update_count].second;
                    popCnt_UM++;
                }

                // for(int i=0; i<1; ){
                //     //if(parallel_UM_W_status.Update_Manager_is_awake==true)
                //         //printf("\t\t\t\t\t\t\tUM1\n");
                //     check_pop_single_task=update_manager_queue.try_pop(single_task);
                //     usleep(50);
                //     if(check_pop_single_task==true){
                //         break;
                //     }
                //     //printf("check system status cores: %d, main_q size: %d\n", parallel_UM_W_status.Available_cores, main_queue.size());
                //     if((parallel_UM_W_status.Available_cores==config.worker_num)&&(main_queue.size()<=0)){
                //         break;
                //     }
                //     if(parallel_UM_W_status.TaskFinish==true){
                //         break;
                //     }   
                // }

                {
                    if(single_task.type==DUPDATE){
                        //printf("\t\t\t\t\t\t\tGet an update\n");
                        update_number_in_round++;
                        UpdateManager_batch_recorder.update_start[UpdateManager_batch_recorder.tail]=single_task.update_start;
                        UpdateManager_batch_recorder.update_end[UpdateManager_batch_recorder.tail]=single_task.update_end;
                        UpdateManager_batch_recorder.tail+=1;
                    }
                    else{
                        //printf("\t\t\t\t\t\t\tGet a query\n");
                        query_number_in_round++;
                    }
                    printf("\t\t\tStart Shuffling \n");
                    ShufflingController_batch_order_queue(newGraph, single_task, orderedList_update, orderedList_query, orderedList_storage, query_number_in_round, update_number_in_round, inacc_map_head, popCnt, theta, num_total_workers);
                }

                totalOrderedListSize=orderedList_update.size();
                totalQueryListSize=orderedList_query.size();
                totalStorageListSize=orderedList_storage.size();

                //if(((parallel_UM_W_status.Available_cores==parallel_UM_W_status.Total_cores)&&((int)main_queue.size()<=0)) && (totalOrderedListSize+totalStorageListSize)>0){
                if(popCnt_UM>=config.query_size+config.update_size){
                    while(!((parallel_UM_W_status.Available_cores==parallel_UM_W_status.Total_cores)&&((int)main_queue_PFS.size()<=0))){//
                        usleep(10);
                        if(parallel_UM_W_status.TaskFinish==true){
                            break;
                        }
                    }
                    printf("\t\t\t====================================\n");
                    printf("\t\t\tClean the Workers_batch_recorder: \n");
                    for(int i=0; i<Workers_batch_recorder.tail; i++){
                        update_u=Workers_batch_recorder.update_start[i];
                        update_v=Workers_batch_recorder.update_end[i];
                        Workers_batch_recorder.update_start[i]=0;
                        Workers_batch_recorder.update_end[i]=0;
                        Workers_batch_recorder.node_type[update_u]=0;
                        Workers_batch_recorder.node_type[update_v]=0;
                    }
                    Workers_batch_recorder.tail=0;
                    printf("\t\t\tAdd the new updates to Workers_batch_recorder: \n");
                    for(int i=0; i<UpdateManager_batch_recorder.tail; i++){
                        update_u=UpdateManager_batch_recorder.update_start[i];
                        update_v=UpdateManager_batch_recorder.update_end[i];
                        Workers_batch_recorder.update_start[Workers_batch_recorder.tail]=update_u;
                        Workers_batch_recorder.update_end[Workers_batch_recorder.tail]=update_v;
                        Workers_batch_recorder.tail+=1;
                        Workers_batch_recorder.node_type[update_u]=1;
                        Workers_batch_recorder.node_type[update_v]=1;
                    }
                    UpdateManager_batch_recorder.tail=0;
                    printf("\t\t\tCopy the inacc_idx to worker's inacc idx: \n");
                    double copy_start=omp_get_wtime();
                    for(int i=0; i<=inacc_map_head; i++){
                        //printf("Copy %d inacc idx \n", i);
                        inacc_idx_map_worker.inacc[i].assign(inacc_idx_map_UM[i].begin(), inacc_idx_map_UM[i].end());
                    }
                    inacc_idx_map_worker.tail=inacc_map_head;
                    inacc_idx_map_UM[0].clear();
                    inacc_idx_map_UM[0].assign(newGraph.getNumOfVertices(), 1);
                    double copy_end=omp_get_wtime();
                    //printf("check copy time: %.8f\n", copy_end-copy_start);
                    inacc_map_head=0;
                    //printf("====================================\n");
                    
                    int addSize;
                    addSize=totalQueryListSize;
                    std::cout<<"\t\t\tIt is time to add"<<addSize<<"query tasks"<<endl;
                    for(int i=0; i<addSize; i++){
                        std::cout << i+1 << "\t\t\t: check workload type is----  " << orderedList_query.front().type << endl;
                        if(orderedList_query.front().type==DQUERY){
                            printf("Query type is: %d\n", orderedList_query.front().move_BehindOrAfter);
                        }
                        main_queue_PFS.push(orderedList_query.front());
                        orderedList_query.pop_front();
                    }
                    addSize=totalOrderedListSize;
                    std::cout<<"\t\t\tIt is time to add"<<addSize<<"update tasks"<<endl;
                    for(int i=0; i<addSize-1; i++){
                        std::cout << i+1 << "\t\t\t: check workload type is----  " << orderedList_update.front().type << endl;
                        single_task_for_push=orderedList_update.front();
                        single_task_for_push.idx_number=1;
                        main_queue_PFS.push(single_task_for_push);
                        orderedList_update.pop_front();
                    }
                    if(addSize>0){
                        std::cout << addSize << "\t\t\t: check last workload type is----  " << orderedList_update.front().type << endl;
                        single_task_for_push=orderedList_update.front();
                        single_task_for_push.idx_number=0; //worker update inacc according to update manager's result when meet this update
                        main_queue_PFS.push(single_task_for_push);
                        orderedList_update.pop_front();
                    }
                    parallel_UM_W_status.Worker_needs_workload=false;
                    query_number_in_round=0;
                    update_number_in_round=0;
                    //printf("====================================\n");
                }
                else if(query_number_in_round>=num_total_workers){
                    printf("\t\t\t====================================\n");
                    while(!((parallel_UM_W_status.Available_cores==parallel_UM_W_status.Total_cores)&&((int)main_queue_PFS.size()<=0))){//&&((int)main_queue.size()<=0)
                        usleep(10);
                        if(parallel_UM_W_status.TaskFinish==true){
                            break;
                        }
                    }
                    if(parallel_UM_W_status.TaskFinish==true){
                        break;
                    }
                    printf("\t\t\tClean the Workers_batch_recorder: \n");
                    for(int i=0; i<Workers_batch_recorder.tail; i++){
                        update_u=Workers_batch_recorder.update_start[i];
                        update_v=Workers_batch_recorder.update_end[i];
                        Workers_batch_recorder.update_start[i]=0;
                        Workers_batch_recorder.update_end[i]=0;
                        Workers_batch_recorder.node_type[update_u]=0;
                        Workers_batch_recorder.node_type[update_v]=0;
                    }
                    Workers_batch_recorder.tail=0;
                    //printf("Add the new updates to Workers_batch_recorder: \n");
                    for(int i=0; i<UpdateManager_batch_recorder.tail; i++){
                        update_u=UpdateManager_batch_recorder.update_start[i];
                        update_v=UpdateManager_batch_recorder.update_end[i];
                        Workers_batch_recorder.update_start[Workers_batch_recorder.tail]=update_u;
                        Workers_batch_recorder.update_end[Workers_batch_recorder.tail]=update_v;
                        Workers_batch_recorder.tail+=1;
                        Workers_batch_recorder.node_type[update_u]=1;
                        Workers_batch_recorder.node_type[update_v]=1;
                    }
                    UpdateManager_batch_recorder.tail=0;
                    //printf("Copy the inacc_idx to worker's inacc idx: \n");
                    double copy_start=omp_get_wtime();
                    for(int i=0; i<=inacc_map_head; i++){
                        //printf("Copy %d inacc idx \n", i);
                        inacc_idx_map_worker.inacc[i].assign(inacc_idx_map_UM[i].begin(), inacc_idx_map_UM[i].end());
                    }
                    double copy_end=omp_get_wtime();
                    //printf("check copy time: %.8f\n", copy_end-copy_start);
                    inacc_idx_map_worker.tail=inacc_map_head;
                    inacc_idx_map_UM[0].clear();
                    inacc_idx_map_UM[0].assign(newGraph.getNumOfVertices(), 1);
                    inacc_map_head=0;
                    //printf("====================================\n");
                    
                    int addSize;
                    addSize=totalQueryListSize;
                    std::cout<<"\t\t\tIt is time to add"<<addSize<<"query tasks"<<endl;
                    for(int i=0; i<addSize; i++){
                        std::cout << i+1 << "\t\t\t: check workload type is----  " << orderedList_query.front().type << endl;
                        if(orderedList_query.front().type==DQUERY){
                            printf("Query type is: %d\n", orderedList_query.front().move_BehindOrAfter);
                        }
                        main_queue_PFS.push(orderedList_query.front());
                        orderedList_query.pop_front();
                    }
                    addSize=totalOrderedListSize;
                    std::cout<<"\t\t\tIt is time to add"<<addSize<<"update tasks"<<endl;
                    for(int i=0; i<addSize-1; i++){
                        std::cout << i+1 << "\t\t\t: check workload type is----  " << orderedList_update.front().type << endl;
                        single_task_for_push=orderedList_update.front();
                        single_task_for_push.idx_number=1;
                        main_queue_PFS.push(single_task_for_push);
                        orderedList_update.pop_front();
                    }
                    if(addSize>0){
                        std::cout << addSize << "\t\t\t: check last workload type is----  " << orderedList_update.front().type << endl;
                        single_task_for_push=orderedList_update.front();
                        single_task_for_push.idx_number=0; //worker update inacc according to update manager's result when meet this update
                        main_queue_PFS.push(single_task_for_push);
                        orderedList_update.pop_front();
                    }
                    parallel_UM_W_status.Worker_needs_workload=false;
                    query_number_in_round=0;
                    update_number_in_round=0;
                    //printf("====================================\n");
                    UpdateManager_batch_recorder.tail=0;
                    printf("\t\t\t====================================\n");
                    
                }
            }
        }));
    }
//---------------------------------------Worker Part--------------------------------
    
    for (uint64_t thread_i = numProducerThreads+numUpdateManagerThreds; thread_i < numConsumers+numUpdateManagerThreds+numProducerThreads; ++thread_i) {
        threads.push_back(std::thread([&, thread_i]() {
            while (!start_flag){
                sleep(10);
            }
            // thread_set.insert(std::this_thread::get_id());
            Agenda_class agenda_worker(newGraph, config.epsilon, (thread_i-numProducerThreads-numUpdateManagerThreds));
            int temp_pop_cnt;
            //DY_worktask single_task;
            DY_worktask single_task;
            bool check_pop_single_task=false;
            while(true) {
                pop_queue_mtx.lock();

                if (popCnt>=config.total_task_size) {
                    parallel_UM_W_status.TaskFinish=true;
                    pop_queue_mtx.unlock();
                    //std::cout<<"finish!"<<endl;
                    break;
                }
                main_queue_PFS.pop(single_task);
                // int workload_type=dynamic_workload[dynamic_workload_tail];
                // dynamic_workload_tail++;
                
                parallel_UM_W_status.lck.lock();
                parallel_UM_W_status.Available_cores-=1;
                printf("Th %d start\n", thread_i-numProducerThreads-numUpdateManagerThreds);
                parallel_UM_W_status.lck.unlock();
                //std::cout << "Worker" << i<<": "<< std::this_thread::get_id()<< " get " << temp_pop_cnt << "^th item doing " << task_type <<"." << std::endl;
                
                temp_pop_cnt = popCnt;
                popCnt++;
                //printf("Th %d 1\n", thread_i-numProducerThreads-numUpdateManagerThreds);
                //pop_queue_mtx.unlock();

                if(single_task.type == DQUERY){
                    query_count++;
                    printf("Th %d Q1.1\n", thread_i-numProducerThreads-numUpdateManagerThreds);
                    read_mtx.lock();
                    if (++readCnt==1){
                        //write_mtx.lock();
                    }
                    read_mtx.unlock();
                    pop_queue_mtx.unlock();
                } 
                else if(single_task.type == DUPDATE) {
                    update_count++;                    
                    //pop_queue_mtx.lock();
                    printf("Th %d U1.2\n", thread_i-numProducerThreads-numUpdateManagerThreds);
                    for(int i=0; i<1; ){
                        if(readCnt==0){
                            break;
                        }
                        usleep(10);
                    }
                    write_mtx.lock();
                    printf("Th %d U1.3\n", thread_i-numProducerThreads-numUpdateManagerThreds);
                }
                printf("Th %d 2\n", thread_i-numProducerThreads-numUpdateManagerThreds);
                //std::cout << "Consumer thread " << i<<": "<< std::this_thread::get_id()<< " is consuming the " << temp_pop_cnt << "^th item doing " << task_type <<"." << std::endl;
                
                ConsumeItem_ErrorPush(newGraph,  single_task, sequence_start, agenda_worker, numConsumers, response_time_total, query_processing_time_total);

                printf("Th %d 3\n", thread_i-numProducerThreads-numUpdateManagerThreds);
                parallel_UM_W_status.lck.lock();
                parallel_UM_W_status.Available_cores+=1;
                parallel_UM_W_status.lck.unlock();
                //printf("NumOps: %d PopCnt: %d\n", numOps, popCnt);
                printf("Th %d, check system status cores: %d, readCNT:%d\n",thread_i-numProducerThreads-numUpdateManagerThreds ,parallel_UM_W_status.Available_cores, readCnt);
                
                if(single_task.type == DQUERY){
                    read_mtx.lock();
                    if (--readCnt==0){
                        //write_mtx.unlock();
                    }
                    read_mtx.unlock();
                } else if(single_task.type == DUPDATE) {
                    pop_queue_mtx.unlock();
                    write_mtx.unlock();
                }
            }
            printf("Thread: %d, finish!\n", thread_i-numProducerThreads-numUpdateManagerThreds);
        }));
    }
    
    start_flag = true;
    for (auto &thread : threads) {
        thread.join();
    }
    
    sequence_time_total=omp_get_wtime()-sequence_start;
    //-------------------------------------------------------------------------------------------
    ofstream query_time_out_file;
    string folder_path="./";
    folder_path+=config.graph_alias;
    folder_path+="/";
    boost::filesystem::create_directories(folder_path);    
    string s="AgendaParallelShuffling";
    s+="_QueryRate_";
    s+=to_string(config.query_arrRate).substr(0, 5);
    s+="_UpdateRate_";
    s+=to_string(config.update_arrRate).substr(0, 5);
    s+="_ParaTime_";
    s+=to_string((int)(config.queue_time));
    s+="_WorkerNum_";
    s+=to_string((int)(config.total_worker_num));
    s+="_PFS_";
    s+=to_string((int)(config.beta_PFS));
    s+="_0404.txt";
    string final_path_txt=folder_path+s;
    query_time_out_file.open(final_path_txt);
    if (!query_time_out_file.is_open()) // check the file 
        cout << "cannot open file." << endl;
    query_time_out_file<<sequence_time_total<<endl;
    query_time_out_file<<response_time_total<<endl;
    query_time_out_file<<query_processing_time_total<<endl;
    cout<<"trw:"<<num_total_rw<<endl;
	cout<<"hrw:"<<num_hit_idx<<endl;
}

void dynamic_ssquery(Graph& graph){
	vector<int> queries;
    load_ss_query(queries);
    INFO(queries.size());
    unsigned int query_size = queries.size();
    query_size = min( query_size, config.query_size );
    int used_counter=0;
	ppr.init_keys(graph.n);
	vector<pair<int,int>> updates;
	load_update(updates);
	INFO(updates.size());
	
	bool IF_OUTPUT=false;
	int query_count=0;
	int update_count=0;
	int test_k=0;
	if(config.algo == FORA){ //fora
        fora_setting(graph.n, graph.m);
        display_setting();

        fwd_idx.first.initialize(graph.n);
        fwd_idx.second.initialize(graph.n);
		
        for(int i=0; i<dynamic_workload.size(); i++){
            cout<<"----------------------------------"<<endl;
			cout<<"operation "<<i<<endl;
			Timer timer(0);
            /*
			if(i!=(dynamic_workload.size()-1)){
				if(dynamic_workload[i+1]!=DQUERY&&dynamic_workload[i]!=DQUERY){
					continue;
				}
			}
            */
            if(dynamic_workload[i]==DUPDATE){
				int u,v;
				u=updates[update_count].first;
				v=updates[update_count].second;
				update_count++;
                update_graph(graph, u, v);
				
				if(!config.exact&&config.with_rw_idx){
                    if(config.alter_idx == 0){
					    rebuild_idx(graph);
                    }
                    else{
                        rebuild_idx_vldb2010(graph, u, v, 1);
                    }
                    
				}
			}else if(dynamic_workload[i]==DQUERY){
				fora_query_basic(queries[query_count++], graph);
			}
        }
		Timer::show();
		ofstream result_file;
		result_file.open("result/"+config.graph_alias+"/fora.txt");
		result_file<<config.check_size<<endl;
        if(config.check_from!=0){
            INFO(config.check_from);
            query_count=config.check_from;
        }
		for(int i=0; i<config.check_size; i++){
			cerr<<i;
			fora_query_basic(queries[query_count++], graph);
			output_imap(ppr, result_file, test_k);
		}
		cout<<endl;
	}
	else if(config.algo == BATON){ //fora
        fora_setting(graph.n, graph.m);
        display_setting();

        fwd_idx.first.initialize(graph.n);
        fwd_idx.second.initialize(graph.n);
		
        for(int i=0; i<dynamic_workload.size(); i++){
            cout<<"----------------------------------"<<endl;
			cout<<"operation "<<i<<endl;
			Timer timer(0);
            if(dynamic_workload[i]==DUPDATE){
				Timer timer(REBUILD_INDEX);
				int u,v;
                bool is_insert = true;
                u=updates[update_count].first;
                v=updates[update_count].second;
                update_count++;
                update_graph(graph, u, v);
                if(!config.exact&&config.with_rw_idx){
                    if(config.alter_idx == 0){
					    rebuild_idx(graph);
                    }
                    else{
                        rebuild_idx_vldb2010(graph, u, v, 1);
                    }
                    
				}
			}else if(dynamic_workload[i]==DQUERY){
				if(!config.exact){
					fora_query_basic(queries[query_count++], graph);
				}else{
					query_count++;
				}
			}
			
			
        }
		Timer::show();
		ofstream result_file;
		if(config.check_size>0){
			if(config.with_rw_idx){
				Timer timer(REBUILD_INDEX);
				rebuild_idx(graph);
			}
			if(config.exact)
				result_file.open("result/"+config.graph_alias+"/exact.txt");
			else
				result_file.open("result/"+config.graph_alias+"/baton.txt");
			result_file<<config.check_size<<endl;
		}
        if(config.check_from!=0){
            query_count=config.check_from;
        }
		for(int i=0; i<config.check_size; i++){
			cerr<<i;
			if(config.power_iteration&&config.exact){
				unordered_map<int, double> map_ppr;
				{
					Timer timer(PI_QUERY);
					fwd_power_iteration(graph, queries[query_count++], map_ppr);
				}
				//Timer::show();
				vector<pair<int ,double>> temp_top_ppr;
				temp_top_ppr.clear();
				temp_top_ppr.resize(map_ppr.size());
				partial_sort_copy(map_ppr.begin(), map_ppr.end(), temp_top_ppr.begin(), temp_top_ppr.end(), 
					[](pair<int, double> const& l, pair<int, double> const& r){return l.second > r.second;});
				int non_zero_counter=0;
				for(long j=0; j<temp_top_ppr.size(); j++){
					if(temp_top_ppr[j].second>0)
						non_zero_counter++;
				}
				result_file << non_zero_counter << endl;
				for(int j=0; j< non_zero_counter; j++){
					result_file << j << "\t" << temp_top_ppr[j].first << "\t" << temp_top_ppr[j].second << endl;
				}
			}
			else {
				fora_query_basic(queries[query_count++], graph);
				output_imap(ppr, result_file, test_k);
			}
		}
		cout<<endl;
    }
	else if(config.algo == PARTUP){
	
        fora_setting(graph.n, graph.m);
        display_setting();
		
		reverse_idx.first.initialize(graph.n);
        reverse_idx.second.initialize(graph.n);
		fwd_idx.first.initialize(graph.n);
        fwd_idx.second.initialize(graph.n);
		
		int T_count=0;
		double T=16.0;
		double rsum_bound=graph.m*config.epsilon*config.epsilon/4/graph.n/(2*config.epsilon/3+2)/config.alpha/log(2/config.pfail);
		INFO(rsum_bound);
        for(int i=0; i<dynamic_workload.size(); i++){
			Timer timer(0);
            if(dynamic_workload[i]==DUPDATE){
				int u,v;
				u=updates[update_count].first;
				v=updates[update_count].second;
				update_count++;
				double errorlimit=config.epsilon*1.0/graph.n/config.alpha*graph.g[u].size()/T/rsum_bound;
				double epsrate=0.5;
				double sigma=0.5;
				
				if(T_count<T-4){
					
					if(errorlimit==0){
						continue;
					}
					
					INFO(errorlimit*epsrate);
					
					{
						Timer timer(11);
						reverse_push(u, graph, errorlimit*epsrate, 1);
					}
                    update_graph(graph, u, v);
					{
						Timer timer(12);
						if(reverse_idx.first.occur.m_num>1+graph.gr[u].size()){
							T_count++;
							for(long j=0; j<reverse_idx.first.occur.m_num; j++){
								long id = reverse_idx.first.occur[j];
								//cout<<"id: "<<id<<"  reserve: "<<reverse_idx.first[id]<<" residue "<<reverse_idx.second[id]<<endl;
								if(reverse_idx.first[id]>errorlimit*(1-epsrate)){
									update_idx(graph, id);
								}
							}
						}else{
							for(long j=0; j<reverse_idx.first.occur.m_num; j++){
								long id = reverse_idx.first.occur[j];
									update_idx(graph, id);
							}
						}
					}
				}else{
					Timer timer(REBUILD_INDEX);
					T_count=0;
					update_graph(graph, u, v);
					rebuild_idx(graph);
				}
				
				
			}else if(dynamic_workload[i]==DQUERY){
				fora_query_dynamic(queries[query_count++], graph, T_count/T);
			}
        }
		if(config.check_size>0){
			Timer::show();
			ofstream result_file;
			result_file.open("result/"+config.graph_alias+"/partup.txt");
			result_file<<config.check_size<<endl;
			for(int i=0; i<config.check_size; i++){
				cerr<<i;
				fora_query_dynamic(queries[query_count++], graph,  T_count/T);
				output_imap(ppr, result_file, test_k);
			}
			cout<<endl;
		}
    }
    else if(config.algo == LAZYUP){
        fora_setting(graph.n, graph.m);
        display_setting();
		fwd_idx.first.initialize(graph.n);
        fwd_idx.second.initialize(graph.n);
        reverse_idx.first.initialize(graph.n);
        reverse_idx.second.initialize(graph.n);
		inacc_idx.reserve(graph.n);
		inacc_idx.assign(graph.n, 1);
		double theta=config.theta;
        for(int i=0; i<dynamic_workload.size(); i++){
            cout<<"----------------------------------"<<endl;
			cout<<"operation "<<i<<endl;
			Timer timer(0);
            if(dynamic_workload[i]==DUPDATE){
				int u,v;
				
				u=updates[update_count].first;
				v=updates[update_count].second;
				update_count++;
				//double errorlimit=config.epsilon*1.0/graph.n/(1-config.alpha)*config.alpha*graph.g[u].size();
				double errorlimit=double(std::max(1,(int)(graph.g[u].size())))/graph.n;
				double epsrate=1;
                errorlimit = errorlimit * config.r_b_scale;
				if(errorlimit==0){
					continue;
				}
				INFO(errorlimit*epsrate);
				{
					Timer timer(11);
					reverse_push(u, graph, errorlimit*epsrate, 1);
				}
				update_graph(graph, u, v);
				
				{
					Timer timer(12);	
					//for(long j=0; j<reverse_idx.first.occur.m_num; j++){
                    for(long j=0; j<graph.n; j++){
						//long id = reverse_idx.first.occur[j];
                        long id = j;
                        double pmin;
                        if(reverse_idx.first.exist(id))
						    pmin=min((reverse_idx.first[id]+errorlimit*epsrate)*(1-config.alpha)/config.alpha,1.0);
                        else
                            pmin=min((errorlimit*epsrate)*(1-config.alpha)/config.alpha,1.0);
                        //double pmin=(reverse_idx.first[id]+errorlimit*epsrate)/config.alpha;
						inacc_idx[id]*=(1-pmin/graph.g[u].size());
                        //inacc_idx[id]-=pmin/graph.g[u].size();
					}
				}
			}else if(dynamic_workload[i]==DQUERY){
				fora_query_lazy_dynamic(queries[query_count++], graph, theta);
			}
        }
		if(config.check_size>0){
			Timer::show();
			ofstream result_file;
			result_file.open("result/"+config.graph_alias+"/lazyup.txt");
			result_file<<config.check_size<<endl;

            if(config.check_from!=0){
                INFO(config.check_from);
                query_count=config.check_from;
            }
			for(int i=0; i<config.check_size; i++){
				cerr<<i;
				fora_query_lazy_dynamic(queries[query_count++], graph, theta);
				output_imap(ppr, result_file, test_k);
			}
			cout<<endl;
		}
    }
	else if(config.algo == RESACC){ //fora
        fora_setting(graph.n, graph.m);
        display_setting();

        fwd_idx.first.initialize(graph.n);
        fwd_idx.second.initialize(graph.n);
		
        for(int i=0; i<dynamic_workload.size(); i++){
            cout<<"----------------------------------"<<endl;
			cout<<"operation "<<i<<endl;
			Timer timer(0);
            if(dynamic_workload[i]==DUPDATE){
				int u,v;
				u=updates[update_count].first;
				v=updates[update_count].second;
				update_count++;
				update_graph(graph, u, v);
			}else if(dynamic_workload[i]==DQUERY){
				resacc_query(queries[query_count++], graph);
			}
        }
		Timer::show();
		ofstream result_file;
		result_file.open("result/"+config.graph_alias+"/resacc.txt");
		result_file<<config.check_size<<endl;
        if(config.check_from!=0){
                INFO(config.check_from);
                query_count=config.check_from;
            }
		for(int i=0; i<config.check_size; i++){
			cerr<<i;
			resacc_query(queries[query_count++], graph);
			output_imap(ppr, result_file, test_k);
		}
		cout<<endl;
	}
	set_result(graph, used_counter, query_size);
	cout<<"trw:"<<num_total_rw<<endl;
	cout<<"hrw:"<<num_hit_idx<<endl;
}
#endif //FORA_QUERY_H
