for k in 3 4 6 10
do
        for beta in 0.1 0.2 0.3
        do
                python rpclk.py --k $k --beta $beta
        done
done