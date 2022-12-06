function [Y,obj,Wv] = DGMC(X,groundtruth,graph,k_nn_num,lambda,r,s)
classnum = max(groundtruth);
viewnum = length(X);   
n = length(groundtruth); 
[num,~] = size(X{1}); 
Fv = cell(1,viewnum); 
Av_rep = zeros(num); 
X = X';

for i = 1 :viewnum
    for  j = 1:n
         X{i}(j,:) = ( X{i}(j,:) - mean( X{i}(j,:) ) ) / std( X{i}(j,:) );
    end
end

options.NeighborMode = 'KNN';
options.k = k_nn_num; 
options.WeightMode = 'HeatKernel';
for v = 1:viewnum
    Xv = X{v};
    if graph == 1
        Av = constructW(Xv,options);
    else
        Av = constructW_PKN(Xv',k_nn_num);
    end
    Av_rep = Av + Av_rep;   
    Lv = Ls(Av);
    temp = eig1(full(Lv),classnum+1,0);
    Fv{v} = temp(:,2:classnum+1);
    Fv{v} = Fv{v}./repmat(sqrt(sum(Fv{v}.^2,2)),1,classnum);
end

Av_rep = 1/viewnum*Av_rep;
L_rep = Ls(Av_rep);
Y_rep = eig1(L_rep,classnum+1,0);
P = Y_rep(:,2:classnum+1);            
P = P./repmat(sqrt(sum(P.^2,2)),1,classnum); 

R = eye(classnum);

NITER = 30;
changed = zeros(NITER,1);
I = eye(num);
e = ones(num,1);
H = I-(e*e')/num;

obj = [];
for iter = 1:NITER
       
G = P*R;
[PR,g] = max(G,[],2);
Y = TransformL(g,classnum);
[~,ind] = sort(PR);
zg = diag(Y'*G);
zz = diag(Y'*Y);
    
    for it = 1:10
        converged = 0;
        for i = 1:num
            N1 = zg' + G(ind(i),:).*(1-Y(ind(i),:));
            DE1 = zz' + (1-Y(ind(i),:));
            
            N2 = zg' - G(ind(i),:).*Y(ind(i),:);
            DE2 = zz' - Y(ind(i),:);
            
            [~,id1] = max(N1./sqrt(DE1)-N2./sqrt(DE2));
            id0 = find(Y(ind(i),:)==1);
            
            if id1 ~= id0
                Y(ind(i),:) = 0;
                Y(ind(i),id1) = 1;
                zg(id0) = zg(id0) - G(ind(i),id0);
                zg(id1) = zg(id1) + G(ind(i),id1);
                zz(id0) = zz(id0) - 1;
                zz(id1) = zz(id1) + 1;
                converged = converged + 1;
            end
        end
        
        if converged == 0
            break;
        end
    end

    changed(iter) = it; 
  
for v = 1:viewnum
    Wv(v) = r*(trace(H*(P*P')*H*Fv{v}*Fv{v}'))^(r-1); 
end 

Z = Y*(Y'*Y)^(-0.5); 
    
for v = 1:viewnum
    [tmp_u, ~, tmp_v] = svd(P'*Z);
    R = tmp_u * tmp_v';
end 

temp = zeros(num,num);
for v = 1:viewnum
    temp = temp + Wv(v)*Fv{v}*Fv{v}';
end
A = H*temp*H;
B = Z*R';
P = GPI(A,B,lambda,s,P);    
    
tem = 0;
for v = 1:viewnum
    tem = tem + (trace(H*(P*P')*H*Fv{v}*Fv{v}'))^r; 
end
ob = tem - lambda*(norm(Z - P*R, 'fro'))^2;
obj = [obj; ob];

end

[~, Y] = max(Y,[],2);

end

function L0 = Ls(A)
    S0 = A;
    S10 = (S0+S0')/2;
    D10 = diag(sum(S10));
    L0 = D10 - S10;
end