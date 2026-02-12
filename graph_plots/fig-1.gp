# Terminal & output - matching your large-scale publication format
set terminal pdfcairo size 17in,10in enhanced font "Verdana,35"
set output 'fig-1.pdf'

# Formatting
# set title "Comparison of BFS, PR_RST, and GConn Metrics" font ",30"
set ylabel "Execution Time [in ms]" font "Verdana,45"
set xlabel "Graph Dataset" font "Verdana,45"
set grid y lw 2

# Remove top and right mirror tics
set xtics nomirror rotate by 45 right font "Verdana,40"
set ytics nomirror font "Verdana,45"
unset mxtics
unset mytics

# Histogram Style - Cluster of 3 bars per dataset
set style data histograms
set style histogram cluster gap 1
set style fill pattern border -1  # Using patterns for research paper consistency
set boxwidth 1 relative

# Log Scale is essential for this range (0.9 to 31,000)
set logscale y
set yrange [0.5:*]

# Legend
set key top left font "Verdana,40"
set rmargin 15

# Data Block
$GraphData << EOD
# graph           vertices  edges       bfs     pr_rst   gconn
WB      685230    13300290    69      11       1.02691
AS        1696415   22192106    47      21       2.16883
HT    456626    25017136    13      10       0.975872
CD      540486    30491458    5       12       1.00246
SO  2601977   56414194    2236    187      5.51834
RU          23947347  57708624    552     3966     51.2152
LJ  4847571   85706224    188     410      9.77594
K20       1048576   89745562    14352   49       15.7932
EU        50912018  108109320   1487    10691    114.142
K21       2097152   183188180   31712   234      10.9765
CO         3072441   234370166   35      46       6.28336
UK           18520486  523651232   2565    2597     45.2002
EOD

# Plotting columns 4 (bfs), 5 (pr_rst), and 6 (gconn)
# xtic(1) uses the graph name as the label
plot $GraphData using 4:xtic(1) title "BFS"     lc rgb '#1f77b4' fill pattern 2, \
     ''         using 5           title "PR RST"  lc rgb '#d62728' fill pattern 4, \
     ''         using 6           title "GConn"   lc rgb '#2ca02c' fill pattern 8