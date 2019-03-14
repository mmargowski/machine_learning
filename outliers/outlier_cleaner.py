#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []

    for i in range(90):
        error = abs(predictions[i] - net_worths[i])
        cleaned_data.append(tuple((ages[i],net_worths[i], error)))

    cleaned_data = sorted(cleaned_data, key= lambda x: x[2])
    del cleaned_data[-9:]
    return cleaned_data

