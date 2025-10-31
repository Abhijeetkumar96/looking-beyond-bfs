#ifndef MULTICORE_BFS_H
#define MULTICORE_BFS_H

#define ALPHA 15.0
#define BETA 24.0

void multicore_bfs(int num_verts, long totalEdges, int* out_array, long* out_degree_list, int root, int* parents, int* levels);

#endif // MULTICORE_BFS_H