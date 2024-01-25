function [x_hat] = Detect_B_PIC_DSC(xpool, y, H, noiseLevel, iter_num)

    Ht = H' ; 
    var_Noise = noiseLevel;
    [~,U] = size(H);


    x_app = zeros(U,1);
    Var_DSC_prev = 0;
    W = ((eye(U)+1) - eye(U).*2);
    inv_H =  inv(Ht*H);
    invD_H= inv(diag(diag(Ht*H)));
    
    for t = 0:iter_num-1
        % BSO
        if t==0
            zigma_A  = inv_H;
        else
            zigma_A = invD_H;
        end
        item1 = Ht*y;
        item2 = (W.*(Ht*H))*x_app;
        
        mu_A_B = zigma_A *( Ht*y - (W.*(Ht*H))*x_app);
        zigma_A_B = max(var_Noise .* (diag(inv_H)),eps);
        
        % BSE
        [mu_B, ~] = GaussianEst(mu_A_B, zigma_A_B, xpool);
        
        % DSC
        Var_DSC= (inv_H*(Ht*y - Ht*H*mu_B)).^2 ;
        Var_DSC_tot = max(Var_DSC + Var_DSC_prev,eps);
        if t==0
            x_app=mu_B;
        else
            x_app = ((Var_DSC./Var_DSC_tot).*mu_B_prev) + ((Var_DSC_prev./Var_DSC_tot).*mu_B);
        end    
        
        % Storing values
        Var_DSC_prev = Var_DSC;
        mu_B_prev = mu_B;
        
        %% Calculating x_hat, MSE, and BER
        x_hat = mu_B ; 
        x_var = zigma_A_B;
    end
    
end