
for i in {1..10}
do
   echo "Iteration ${i}"
   mkdir falsify_csvs
   python3.9 analyze.py falsify compositional 5
   python3.9 analyze.py falsify monolithic
   mv falsify_csvs falsify_csvs_${i}
   mv falsify_csvs_${i} halton/
done
