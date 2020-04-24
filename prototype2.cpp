#include <iostream>
#include <vector>
#include <stdio.h>
#include <math.h>

double activation_function(double x) {
  return tanh(x);
}

double derivative_activation_function(double x){
    return (1-(x*x));
}

double update_weight(double w, double a, double d){
    return (w - a * d);
}

double feed_forward(NN_layer n1, NN_layer n2)i{
    
    double zh1 = w1 * x1 + w3 * x2 + w5 * x3 + b1;
    
    double zh2 = w2 * x1 + w4 * x2 + w6 * x3 + b2;
}

class NN_layer 
{ 
    public: 
  
        std::vector<double> nodes; 
        std::vector<double> weights;
        double bias;
  
}; 

int main()
{
    //Step 1
    std::vector<int> inputs (1, 4, 5);

    double t1 = 0.1, t2 = 0.05; 
    double alpha = 0.01;

    NN_layer layer1
        
    for  (int i = 1; i <= 6; i++){
        layer1.nodes.pushback(i/10)
    }
    
    layer1.nodes.pushback(1,4,5)
    layer1.bias = 0.5

    NN_layer layer2 
    for  (int i = 7; i <= 10; i++){
        if (i != 10){
            layer2.nodes.pushback(i/10)
        }
        else{
            layer2.nodes.pushback(0.1)
        }
    }
 
    layer2.bias = 0.5


    //for (int i = 0; i < 100000; i++){
        //Step 2
        double zh1 = w1 * x1 + w3 * x2 + w5 * x3 + b1; 
        double zh2 = w2 * x1 + w4 * x2 + w6 * x3 + b2; 

        double h1 = activation_function(zh1);
        double h2 = activation_function(zh1);

        double zo1 = w7 * h1 + w9 * h2 + b2;
        double zo2 = w8 * h1 + w10 * h2 + b2; 
    
        double o1 = activation_function(zo1);
        double o2 = activation_function(zo2);

        //Step 3
        double E = (0.5)*((o1-t1)*(o1-t1)+(o2-t2)*(o2-t2));
        double dE_do1 = o1 - t1; 
        double dE_do2 = o2 - t2;


        //Step 4
        int dzo1_dw7 = h1; 
        int dzo2_dw8 = h1;
        int dzo1_dw9 = h2;
        int dzo2_dw10 = h2; 
        int dzo1_db2  = 1; 
        int dzo2_db2 = 1;

        double dE_dw7 = (o1-t1)*derivative_activation_function(zo1)*h1;
        double dE_dw8 = (o2-t2)*derivative_activation_function(zo2)*h1;
        double dE_dw9 = (o1-t1)*derivative_activation_function(zo1)*h2;
        double dE_dw10 = (o2-t2)*derivative_activation_function(zo2)*h2;

        double dE_db2 = dE_do1 * derivative_activation_function(zo1) * dzo1_db2 + dE_do2 * derivative_activation_function(zo2) * dzo2_db2;





        double dE_dh1 = dE_do1 * derivative_activation_function(zo1) * w7 + dE_do2 * derivative_activation_function(zo2) * w8;
        double dE_dw1 = dE_dh1 * derivative_activation_function(h1) * x1;
        double dE_dw3 = dE_dh1 * derivative_activation_function(h1) * x2;
        double dE_dw5 = dE_dh1 * derivative_activation_function(h1) * x3;






        double dE_dh2 = dE_do1 * derivative_activation_function(zo1) * w9 + dE_do2 * derivative_activation_function(zo2) * w10;
        double dE_dw2 = dE_dh2 * derivative_activation_function(h2) * x1;
        double dE_dw4 = dE_dh2 * derivative_activation_function(h2) * x2;
        double dE_dw6 = dE_dh2 * derivative_activation_function(h2) * x3;



        double dE_db1 = dE_do1 * derivative_activation_function(zo1) * w7 * derivative_activation_function(h1) * 1 + dE_do2 * derivative_activation_function(zo2) * w10 * derivative_activation_function(h2) * 1;

        w1 = update_weight(w1, alpha, dE_dw1);
        w2 = update_weight(w2, alpha, dE_dw1);
        w3 = update_weight(w3, alpha, dE_dw1);
        w4 = update_weight(w4, alpha, dE_dw1);
        w5 = update_weight(w5, alpha, dE_dw1);
        w6 = update_weight(w6, alpha, dE_dw1);
        w7 = update_weight(w7, alpha, dE_dw1);
        w8 = update_weight(w8, alpha, dE_dw1);
        w9 = update_weight(w9, alpha, dE_dw1);
        w10 = update_weight(w10, alpha, dE_dw1);

        b1 = update_weight(b1, alpha, dE_db1);
        b2 = update_weight(b2, alpha, dE_db2);

        std::cout << "w1: " << w1 << std::endl;
    //}
    /*
    std::cout << "w1: " << w1;
    std::cout << "w2: " << w2;
    std::cout << "w3: " << w3;
    std::cout << "w4: " << w4;
    std::cout << "w5: " << w5;
    std::cout << "w6: " << w6;
    std::cout << "w7: " << w7;
    std::cout << "w8: " << w8;
    std::cout << "w9: " << w9;
    std::cout << "w10: " << w10;
    std::cout << "b1: " << b1;
    std::cout << "b2: " << b2;
    */
}
