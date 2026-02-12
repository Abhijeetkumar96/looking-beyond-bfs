.PHONY: all bfs_adam_polak gconn pr_rst clean

all: bfs_adam_polak gconn pr_rst

bfs_adam_polak:
	$(MAKE) -C bfs_adam_polak

gconn:
	$(MAKE) -C gconn

pr_rst:
	cmake -S PR-RST/PR-RST_min_max_iter -B PR-RST/PR-RST_min_max_iter/build
	cmake --build PR-RST/PR-RST_min_max_iter/build

clean:
	$(MAKE) -C bfs_adam_polak clean
	$(MAKE) -C gconn clean
	@if [ -d PR-RST/PR-RST_min_max_iter/build ]; then \
		$(MAKE) -C PR-RST/PR-RST_min_max_iter/build clean; \
	fi
