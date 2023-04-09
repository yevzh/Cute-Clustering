for method in aic bic vbem
do
        for n_cluster in 3 4 5 6
        do
                for n_samples in 10 20 50 100
                do
                        python GMM.py --method $method --n_cluster $n_cluster --n_samples $n_samples
                done
        done
done