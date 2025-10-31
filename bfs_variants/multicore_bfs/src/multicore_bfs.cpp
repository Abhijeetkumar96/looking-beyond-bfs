#include <omp.h>
#include <iostream>

#include "thread.hpp"
#include "multicore_bfs.hpp"

bool verbose = false; 
bool debug   = false;

void multicore_bfs(int num_verts, long totalEdges, int* out_array, long* out_degree_list, int root, int* parents, int* levels) {
    
    double avg_out_degree = totalEdges/(double)num_verts;

    int* queue = new int[num_verts];
    int* queue_next = new int[num_verts];
    int queue_size = 0;  
    int queue_size_next = 0;

    queue[0] = root;
    queue_size = 1;
    parents[root] = root;
    levels[root] = 0;

    int level = 1;
    double elt, elt2 = 0.0;
    int num_descs = 0;
    int local_num_descs = 0;
    bool use_hybrid = false;
    bool already_switched = false;

    #pragma omp parallel
    {
        int thread_queue[ THREAD_QUEUE_SIZE ];
        int thread_queue_size = 0;

        while (queue_size)
        {

            if (!use_hybrid)
            {
                #pragma omp for schedule(guided) reduction(+:local_num_descs) nowait
                for (int i = 0; i < queue_size; ++i)
                {
                    int vert = queue[i];
                    long out_degree = out_degree_list[vert+1] - out_degree_list[vert];
                    int* outs = &out_array[out_degree_list[vert]];
                    for (long j = 0; j < out_degree; ++j)
                    {      
                        int out = outs[j];
                        if (levels[out] < 0)
                        {
                            levels[out] = level;
                            parents[out] = vert;
                            ++local_num_descs;
                            add_to_queue(thread_queue, thread_queue_size, queue_next, queue_size_next, out);
                        }
                    }
                }
            }
            else
            {
                int prev_level = level - 1;

                #pragma omp for schedule(guided) reduction(+:local_num_descs) nowait
                for (int vert = 0; vert < num_verts; ++vert)
                {
                    if (levels[vert] < 0)
                    {
                        long out_degree = out_degree_list[vert+1] - out_degree_list[vert];
                        int* outs = &out_array[out_degree_list[vert]];
                        for (long j = 0; j < out_degree; ++j)
                        {
                            int out = outs[j];
                            if (levels[out] == prev_level)
                            {
                                levels[vert] = level;
                                parents[vert] = out;
                                ++local_num_descs;
                                add_to_queue(thread_queue, thread_queue_size, queue_next, queue_size_next, vert);
                                break;
                            }
                        }
                    }
                }
            }
    
            empty_queue(thread_queue, thread_queue_size, queue_next, queue_size_next);
            #pragma omp barrier

            #pragma omp single
            {
                if (debug)
                  std::cout << "num_descs: " << num_descs << " local: " << local_num_descs << std::endl;
                num_descs += local_num_descs;

                if (!use_hybrid)
                {  
                    double edges_frontier = (double)local_num_descs * avg_out_degree;
                    double edges_remainder = (double)(num_verts - num_descs) * avg_out_degree;
                    if ((edges_remainder / ALPHA) < edges_frontier && edges_remainder > 0 && !already_switched)
                    {
                        if (debug)
                            std::cout << "\n=======switching to hybrid\n\n";

                        use_hybrid = true;
                    }
                    if (debug)
                        std::cout << "edge_front: " << edges_frontier << ", edge_rem: " << edges_remainder << std::endl;
                }
                else
                {
                    if ( ((double)num_verts / BETA) > local_num_descs  && !already_switched)
                    {
                        if (debug)
                            std::cout << "\n=======switching back\n\n";

                        use_hybrid = false;
                        already_switched = true;
                    }
                }
                local_num_descs = 0;

                queue_size = queue_size_next;
                queue_size_next = 0;
                int* temp = queue;
                queue = queue_next;
                queue_next = temp;
                ++level;
            } // end single
        }
    } // end parallel


    if (debug)
        std::cout <<"Final num desc: " << num_descs << std::endl;
  
    delete [] queue;
    delete [] queue_next;
}