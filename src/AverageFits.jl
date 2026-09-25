module AverageFits

using LinearAlgebra, Statistics, Hungarian, DataStructures, ImageFiltering

export noisefilter, normalizeU!, fitd, matchedfitval, matchedorder, flip2makepos!
export getdata, subtract_baseline, evaluate_fitvalue, matchedWnssda, ssdH
export matchedfitval_clamp
export matched_correlation, evaluate_correlation, wh_correlations

"""
X = noisefilter(filter,X)
filter : :medT, :meanT, :medS
"""
function noisefilter(filter,X)
    if filter == :medT
        X = mapwindow(median!, X, (1,3)) # just for each row
    elseif filter == :meanT
        X = mapwindow(mean, X, (1,3)) # just for each row
    elseif filter == :medS
        rsimg = reshape(X,imgsz...,lengthT)
        rsimgm = mapwindow(median!, rsimg, (3,3,1))
        X = reshape(rsimgm,*(imgsz...),lengthT)
    end
    X
end

function normalizeU!(U,V)
    p = size(U,2)
    for i = 1:p
        nrm = max(eps(eltype(U)),norm(U[:,i]))
        if nrm != 0.
            U[:,i] ./= nrm
            V[i,:] .*= nrm
        end
    end
    U,V
end

fitx(a,b) = (m=sum(a)/length(a); denom=sum(abs2,a.-m); fitx(a,b,denom))
fitx(a,b,denom) = (1-sum(abs2,a-b)/denom)
fitd(a,b) = (na=norm(a); nb=norm(b); fitd(a,b,na,nb))
fitd(a,b,na) = (nb=norm(b); fitd(a,b,na,nb))
fitd(a,b,na,nb) = (denom=na^2+nb^2+2na*nb; (1-sum(abs2,a-b)/denom, denom))
#fitd(a,b,na,nb) = (denom=na^2+nb^2+2na*nb; 1-sum(abs2,a-b)/denom)
fitd(a,b,nbn,na,nb) = (denom=na^2+nb^2+2na*nb; (1-(sum(abs2,a-b)+nbn^2)/denom,denom))
# calfit(a,b) = (dval=fitd(a,b); aval=fitd(a,-b); dval > aval ? (dval, false) : (aval, true))
# calfit(a,b,na) = (dval=fitd(a,b,na); aval=fitd(a,-b,na); dval > aval ? (dval, false) : (aval, true))
ssd(a,b) = sum(abs2,a-b)
nssd(a,b) = (ssd(a,b)/(norm(a)*norm(b)), false)
nssda(a,b) = (ssdval=ssd(a,b); ssaval=ssd(a,-b); nab=(norm(a)*norm(b));
                ssdval < ssaval ? (ssdval/nab, false) : (ssaval/nab, true))
function fiterr(a,b)
    init_x = eltype(a)[1, 0]
    f(x) = norm((x[1].*b.+x[2]).-a)^2
    rst = optimize(f,init_x)
    rst.minimum/length(a), false
end

"""
    matchlist, ssds = matchcomponents(GT, W, errorfn::Function)
Matched the W columns with with those of GT.
GT: ground truch matrix
W: matrix to Compare
errorfn: function used to calculate the error of two vectors
matchlist: list of pair = (column index of GT, column index of W)
ssds: list of ssd for the matched pair columns
"""
function matchWcomponents(GT, W, errorfn::Function) # M X r form
    gtcolnum = size(GT,2); wcolnum = size(W,2)
    costmat = zeros(Float64, gtcolnum, wcolnum)
    invertmat = falses(gtcolnum, wcolnum)
    for i = 1:gtcolnum
        gti = GT[:,i]
        for j = 1:wcolnum
            wj = W[:,j]
            dist, invert = errorfn(gti,wj)
            costmat[i,j] = dist
            invertmat[i,j] = invert
        end
    end
    # globally optimal 1:1 matching (minimizes total dist), instead of greedy nearest-neighbor
    assignment, _ = hungarian(costmat)
    matchlist = Tuple{Int,Int,Bool}[]
    errs = Float64[]
    for i = 1:gtcolnum
        j = assignment[i]
        j == 0 && continue # unmatched (only possible when wcolnum < gtcolnum)
        push!(matchlist,(i,j,invertmat[i,j]))
        push!(errs,costmat[i,j])
    end
    matchlist, errs
end

function matchcomponents(GTW::AbstractArray{T}, GTH::AbstractArray{T}, W::AbstractArray{T}, H::AbstractArray{T};
        clamp=false, iscalunmatched=false, sdsr=1, tdsr=1) where T
    gtcolnum = size(GTW,2); wcolnum = size(W,2)
    costmat = zeros(T, gtcolnum, wcolnum)
    invertmat = falses(gtcolnum, wcolnum)
    for i = 1:gtcolnum
        gtwi = GTW[1:sdsr:end,i]; gthi = GTH[i,1:tdsr:end]; gtxi = gtwi*gthi'
        for j = 1:wcolnum
            wj = W[1:sdsr:end,j]; hj = H[j,1:tdsr:end]; xj = wj*hj'
            clamp && (xj[xj.<0].=0)
            mnssd, invert = nssd(gtxi,xj)
            costmat[i,j] = mnssd
            invertmat[i,j] = invert
        end
    end
    # globally optimal 1:1 matching (minimizes total nssd), instead of greedy nearest-neighbor
    assignment, _ = hungarian(costmat)
    matchlist = Tuple{Int,Int,Bool}[]; ml = Int[]
    mnssds = T[]
    for i = 1:gtcolnum
        j = assignment[i]
        j == 0 && continue
        push!(matchlist,(i,j,invertmat[i,j]))
        push!(ml,j)
        push!(mnssds,costmat[i,j])
    end
    # Calculate unmatched power
    unmatchlist = collect(1:wcolnum)
    filter!(a->a ∉ ml,unmatchlist)
    rerrs = T[]
    if iscalunmatched
        for j in unmatchlist
            wj = W[:,j]; hj = H[j,:]; xj = wj*hj'
            rerr = sum(abs2,xj)
            push!(rerrs,rerr)
        end
    end
    matchlist, mnssds, rerrs
end

function matchedorder(GTW::AbstractArray{T}, GTH::AbstractArray{T}, W::AbstractArray{T}, H::AbstractArray{T},
            noc; weighted=true, clamp=false, iscalunmatched=false, sdsr=1, tdsr=1) where T
    gtcolnum = size(GTW,2); wcolnum = size(W,2)
    gtWsum = dropdims(sum(abs,GTW, dims=2), dims=2); gtHsum = dropdims(sum(abs,GTH, dims=1), dims=1)
    iw = gtWsum.!=0; ih = gtHsum.!=0 # to reduce computation, choose only non-zero rows and column

    fitmat = zeros(T, gtcolnum, wcolnum); denommat = zeros(T, gtcolnum, wcolnum)
    for i = 1:gtcolnum
        gtwi = GTW[iw,i]; gthi = GTH[i,ih]; gtxi = gtwi[1:sdsr:end]*gthi[1:tdsr:end]'; ngtxi = norm(gtxi)
        for j = 1:wcolnum
            wj = W[iw,j]; hj = H[j,ih]; xj = wj[1:sdsr:end]*hj[1:tdsr:end]'
            clamp && (xj[xj.<0].=0)
            fitval, denom = fitd(gtxi,xj,ngtxi)
            fitmat[i,j] = fitval
            denommat[i,j] = weighted ? denom : one(T)
        end
    end
    # globally optimal 1:1 matching maximizing total (weighted) fitval, instead of greedy nearest-neighbor
    weightmat = weighted ? fitmat .* denommat : fitmat
    assignment, _ = hungarian(-weightmat)
    matchlist = Tuple{Int,Int,Bool}[]; ml = Int[]
    fitvals = T[]; denomsum = zero(T)
    for i = 1:gtcolnum
        j = assignment[i]
        j == 0 && continue
        push!(matchlist,(i,j,false))
        push!(ml,j)
        push!(fitvals,fitmat[i,j]*denommat[i,j]) # power weighted fitval
        denomsum += denommat[i,j]
    end
    # Calculate unmatched power
    rerrs = T[]
    if iscalunmatched
        unmatchlist = collect(1:wcolnum)
        filter!(a->a ∉ ml,unmatchlist)
        for j in unmatchlist
            wj = W[iw,j]; hj = H[j,ih]; xj = wj*hj'
            rerr = sum(abs2,xj)
            push!(rerrs,rerr)
        end
    end
    nodr = matchedorder(matchlist, noc)
    nodr, matchlist, fitvals, rerrs, denomsum
end

function fitcomponents_clamp(GTW::AbstractArray{T}, GTH::AbstractArray{T}, W::AbstractArray{T}, H::AbstractArray{T};
            weighted=true, clamp=false, iscalunmatched=false, sdsr=1, tdsr=1) where T
    gtcolnum = size(GTW,2); wcolnum = size(W,2)
    fitmat = zeros(T, gtcolnum, wcolnum); denommat = zeros(T, gtcolnum, wcolnum)
    for i = 1:gtcolnum
        gtwi = GTW[:,i]; gthi = GTH[i,:]; gtxi = gtwi[1:sdsr:end]*gthi[1:tdsr:end]'; ngtxi = norm(gtxi)
        for j = 1:wcolnum
            wj = W[:,j]; hj = H[j,:]; xj = wj[1:sdsr:end]*hj[1:tdsr:end]'
            clamp && (xj[xj.<0].=0)
            fitval, denom = fitd(gtxi,xj,ngtxi)
            fitmat[i,j] = fitval
            denommat[i,j] = weighted ? denom : one(T)
        end
    end
    # globally optimal 1:1 matching maximizing total (weighted) fitval, instead of greedy nearest-neighbor
    weightmat = weighted ? fitmat .* denommat : fitmat
    assignment, _ = hungarian(-weightmat)
    matchlist = Tuple{Int,Int,Bool}[]; ml = Int[]
    fitvals = T[]; denomsum = zero(T)
    for i = 1:gtcolnum
        j = assignment[i]
        j == 0 && continue
        push!(matchlist,(i,j,false))
        push!(ml,j)
        push!(fitvals,fitmat[i,j]*denommat[i,j]) # power weighted fitval
        denomsum += denommat[i,j]
    end
    # Calculate unmatched power
    rerrs = T[]
    if iscalunmatched
        unmatchlist = collect(1:wcolnum)
        filter!(a->a ∉ ml,unmatchlist)
        for j in unmatchlist
            wj = W[:,j]; hj = H[j,:]; xj = wj*hj'
            rerr = sum(abs2,xj)
            push!(rerrs,rerr)
        end
    end
    matchlist, fitvals, rerrs, denomsum
end

# Fast path: gtxi = gtwi*gthi' and xj = wj*hj' are rank-1, so dot(gtxi,xj) = dot(gtwi,wj)*dot(gthi,hj)
# and norm(xj)^2 = norm(wj)^2*norm(hj)^2. This avoids ever forming the m x n outer-product matrices,
# turning the O(gtcolnum*wcolnum*m*n) double loop into O(gtcolnum*wcolnum*(m+n)).
# clamp=true breaks the rank-1 structure (it zeros negative entries of xj elementwise), so that case
# falls back to fitcomponents_clamp.
function fitcomponents(GTW::AbstractArray{T}, GTH::AbstractArray{T}, W::AbstractArray{T}, H::AbstractArray{T};
            weighted=true, clamp=false, iscalunmatched=false, sdsr=1, tdsr=1) where T
    clamp && return fitcomponents_clamp(GTW, GTH, W, H; weighted=weighted, clamp=clamp,
                                         iscalunmatched=iscalunmatched, sdsr=sdsr, tdsr=tdsr)
    gtcolnum = size(GTW,2); wcolnum = size(W,2)
    GTWs = GTW[1:sdsr:end,:]; GTHs = permutedims(GTH[:,1:tdsr:end])
    Ws   = W[1:sdsr:end,:];   Hs   = permutedims(H[:,1:tdsr:end])

    gtwn2 = vec(sum(abs2,GTWs,dims=1)); gthn2 = vec(sum(abs2,GTHs,dims=1))
    wn2   = vec(sum(abs2,Ws,  dims=1)); hn2   = vec(sum(abs2,Hs,  dims=1))
    ngt2  = gtwn2 .* gthn2   # ‖gtxi‖^2, length gtcolnum
    nx2   = wn2 .* hn2       # ‖xj‖^2,   length wcolnum

    cross = (GTWs' * Ws) .* (GTHs' * Hs)   # dot(gtxi,xj), gtcolnum x wcolnum
    diff2 = ngt2 .+ nx2' .- 2 .* cross     # sum(abs2, gtxi-xj)
    ngt = sqrt.(ngt2); nx = sqrt.(nx2)'
    realdenom = (ngt .+ nx).^2             # (‖gtxi‖+‖xj‖)^2, used for fitval regardless of `weighted`
    fitmat = 1 .- diff2 ./ realdenom
    denommat = weighted ? realdenom : ones(T, gtcolnum, wcolnum)

    # globally optimal 1:1 matching maximizing total (weighted) fitval, instead of greedy nearest-neighbor
    weightmat = weighted ? fitmat .* denommat : fitmat
    assignment, _ = hungarian(-weightmat)
    matchlist = Tuple{Int,Int,Bool}[]; ml = Int[]
    fitvals = T[]; denomsum = zero(T)
    for i = 1:gtcolnum
        j = assignment[i]
        j == 0 && continue
        push!(matchlist,(i,j,false))
        push!(ml,j)
        push!(fitvals,fitmat[i,j]*denommat[i,j]) # power weighted fitval
        denomsum += denommat[i,j]
    end
    # Calculate unmatched power (uses full, non-subsampled W,H, matching fitcomponents_clamp)
    rerrs = T[]
    if iscalunmatched
        unmatchlist = collect(1:wcolnum)
        filter!(a->a ∉ ml,unmatchlist)
        for j in unmatchlist
            rerr = sum(abs2,view(W,:,j)) * sum(abs2,view(H,j,:))
            push!(rerrs,rerr)
        end
    end
    matchlist, fitvals, rerrs, denomsum
end

function fitcomponents(X::AbstractArray, GTX::AbstractVector, W::AbstractArray{T}, H::AbstractArray{T};
            weighted=true, clamp=false, ordered=false) where T
    gtcolnum = length(GTX); wcolnum = size(W,2)
    fitvals = T[]; denomsum = zero(T)
    matchlist = Tuple{Int,Int,Bool}[]
    if ordered
        for i = 1:gtcolnum
            # Read bounding box from GTX[i][2] and mask from GTX[i][3]
            # Then, perform masking with mask in the bounding box
            # Ground truth X values of all the outside of the mask are assumed to be zero.
            vecs = vcat(collect.(GTX[i][2])...); idxs = vecs[Bool.(GTX[i][3])] # TODO: make eltype(GTX[i][3]) as Bool
            gtxi = X[idxs,:]; ngtxi = norm(gtxi)
            wi = W[idxs,i]; hi = H[i,:]; xi = wi*hi'
            nxi2 = norm(xi)^2; nxiall2 = norm(W[:,i]*H[i,:]')^2; nxin2 = nxiall2-nxi2
            clamp && (xi[xi.<0].=0)
            fitval, denom = fitd(gtxi,xi,sqrt(nxin2),sqrt(nxiall2),ngtxi)
            push!(fitvals,fitval*denom) # power weighted fitval
            denomsum += denom
        end
    else
        fitmat = zeros(T, gtcolnum, wcolnum); denommat = zeros(T, gtcolnum, wcolnum)
        for i = 1:gtcolnum
            vecs = vcat(collect.(GTX[i][2])...); idxs = vecs[Bool.(GTX[i][3])] # TODO: make eltype(GTX[i][3]) as Bool
            gtxi = X[idxs,:]; ngtxi = norm(gtxi)
            for j = 1:wcolnum
                wj = W[idxs,j]; hj = H[j,:]; xj = wj*hj' # when mask value is 1
                nxjall2 = norm(W[:,j]*H[j,:]')^2; nxjn2 = nxjall2-norm(xj)^2
                clamp && (xj[xj.<0].=0)
                fitval, denom = fitd(gtxi,xj,sqrt(nxjn2),sqrt(nxjall2),ngtxi)
                fitmat[i,j] = fitval
                denommat[i,j] = weighted ? denom : one(T)
            end
        end
        # globally optimal 1:1 matching maximizing total (weighted) fitval, instead of greedy nearest-neighbor
        weightmat = weighted ? fitmat .* denommat : fitmat
        assignment, _ = hungarian(-weightmat)
        for i = 1:gtcolnum
            j = assignment[i]
            j == 0 && continue
            push!(matchlist,(i,j,false))
            push!(fitvals,fitmat[i,j]*denommat[i,j]) # power weighted fitval
            denomsum += denommat[i,j]
        end
    end
    rerrs = T[]
    matchlist, fitvals, rerrs, denomsum
end


function fitcomponents_old(GTW::AbstractArray{T}, GTH::AbstractArray{T}, W::AbstractArray{T}, H::AbstractArray{T};
            weighted=true, clamp=false, iscalunmatched=false, sdsr=1, tdsr=1) where T
    pq = PriorityQueue{Tuple{Int,Int,Bool,T},T}(Base.Order.Reverse) # ((idx,idx,Bool,denom),fitval) Reverse(high->low)
    gtcolnum = size(GTW,2); wcolnum = size(W,2)
    for i = 1:gtcolnum
        gtwi = GTW[:,i]; gthi = GTH[i,:]; gtxi = gtwi[1:sdsr:end]*gthi[1:tdsr:end]'; ngtxi = norm(gtxi)
        for j = 1:wcolnum
            wj = W[:,j]; hj = H[j,:]; xj = wj[1:sdsr:end]*hj[1:tdsr:end]'
            clamp && (xj[xj.<0].=0)
            fitval, denom = fitd(gtxi,xj,ngtxi)
            enqueue!(pq,(i,j,false,weighted ? denom : 1.0), fitval)
        end
    end
    matchlist = Tuple{Int,Int,Bool}[]; ml = Int[]
    fitvals = T[]; denomsum = 0.
    while !isempty(pq)
        p = peek(pq)
        dequeue!(pq)
        found = false
        mllength = length(matchlist)
        for i = 1:mllength
            if p[1][1] == matchlist[i][1] || p[1][2] == matchlist[i][2]
                found = true
                break
            end
        end
        if !found
            push!(matchlist,(p[1][1],p[1][2],p[1][3]))
            push!(ml,p[1][2])
            push!(fitvals,p[2]*p[1][4]) # power weighted fitval
            denomsum += p[1][4]
        end
    end
    # Calculate unmatched power
    rerrs = T[]
    if iscalunmatched
        unmatchlist = collect(1:wcolnum)
        filter!(a->a ∉ ml,unmatchlist)
        gtxi = zeros(T,size(W,1),size(H,2))
        for j in unmatchlist
            wj = W[:,j]; hj = H[j,:]; xj = wj*hj'
            rerr = sum(abs2,xj)
            push!(rerrs,rerr)
        end
    end
    matchlist, fitvals, rerrs, denomsum
end

function fitcomponents_old(X::AbstractArray, GTX::AbstractVector, W::AbstractArray{T}, H::AbstractArray{T};
            weighted=true, clamp=false, ordered=false) where T
    pq = PriorityQueue{Tuple{Int,Int,Bool,T,T}}(Base.Order.Reverse) # (idx,idx,Bool,denom,fitval) Reverse(high->low)
    gtcolnum = length(GTX); wcolnum = size(W,2); allidxs = collect(1:size(W,1))
    fitvals = T[]; denomsum = 0.
    for i = 1:gtcolnum
        # Read bounding box from GTX[i][2] and mask from GTX[i][3]
        # Then, perform masking with mask in the bounding box
        # Ground truth X values of all the outside of the mask are assumed to be zero.
        vecs = vcat(collect.(GTX[i][2])...); idxs = vecs[Bool.(GTX[i][3])] # TODO: make eltype(GTX[i][3]) as Bool
        gtxi = X[idxs,:]; ngtxi = norm(gtxi)
        # calcuate fit value  with each W[:,j]*H[j,:]'
        if ordered
            wi = W[idxs,i]; hi = H[i,:]; xi = wi*hi'
            nxi2 = norm(xi)^2; nxiall2 = norm(W[:,i]*H[i,:]')^2; nxin2 = nxiall2-nxi2
            clamp && (xi[xi.<0].=0)
            fitval, denom = fitd(gtxi,xi,sqrt(nxin2),sqrt(nxiall2),ngtxi) # fitval = fitd(gtxi,xj,nxjn,ngtxi)
            push!(fitvals,fitval*denom) # power weighted fitval
            denomsum += denom
        else
            for j = 1:wcolnum
                wj = W[idxs,j]; hj = H[j,:]; xj = wj*hj' # when mask value is 1
                # Calculate the norm of wjn*hj' which is the X of mask vlaue is 0
                # wjn = W[allidxs[allidxs.∉ [idxs]],j] # when mask value is 0
                # s=0; for w = wjn, h = hj s += (w*h)^2 end; nxjn = sqrt(s)
                nxj2 = norm(xj)^2; nxjall2 = norm(W[:,j]*H[j,:]')^2; nxjn2 = nxjall2-nxj2
                clamp && (xj[xj.<0].=0)
                fitval, denom = fitd(gtxi,xj,sqrt(nxjn2),sqrt(nxjall2),ngtxi) # fitval = fitd(gtxi,xj,nxjn,ngtxi)

                # norm2diff = norm(gtxi-xj)^2; norm2outside = nxjn2
                # norm2nom = norm2diff+norm2outside
                # norm2gt = ngtxi^2; norm2xj = norm(xj)^2; norm2xjall = norm2xj+norm2outside
                # twonorm2gtxjall = 2*ngtxi*sqrt(norm2xjall); norm2denom = norm2gt+norm2xjall+twonorm2gtxjall
                # @show norm2diff, norm2outside, norm2nom
                # @show norm2gt, norm2xj, norm2xjall, twonorm2gtxjall, norm2denom
                # @show fitval, norm2nom/norm2denom, 1-norm2nom/norm2denom

                enqueue!(pq,(i,j,false,weighted ? denom : 1.0), fitval)
            end
        end
    end
    matchlist = Tuple{Int,Int,Bool}[]#; ml = Int[]

    if !ordered
        # find best matched pair (i,j)
        while !isempty(pq)
            p = peek(pq)
            dequeue!(pq)
            found = false
            mllength = length(matchlist)
            for i = 1:mllength
                if p[1][1] == matchlist[i][1] || p[1][2] == matchlist[i][2]
                    found = true
                    break
                end
            end
            if !found
                push!(matchlist,(p[1][1],p[1][2],p[1][3]))
                # push!(ml,p[1][2])
                push!(fitvals,p[2]*p[1][4]) # power weighted fitval
                denomsum += p[1][4]
            end
        end
    else
        # foreach(i->push!(matchlist,(i,i,false)), 1:gtcolnum)
    end
    rerrs = T[]
    # gtindices = map(i->matchlist[i][1],1:gtcolnum)
    # @show fitvals[sortperm(gtindices)], sortperm(gtindices)
    matchlist, fitvals, rerrs, denomsum
end

matchedWnssd(GT,W) = ((ml, nssds) = matchWcomponents(GT, W, nssd); (sum(nssds)/length(nssds), ml, nssds))
matchedWnssda(GT,W) = ((ml, nssdas) = matchWcomponents(GT, W, nssda); (sum(nssdas)/length(nssdas), ml, nssdas))
matchedfitval(GTW, GTH, W, H; weighted=true, clamp=false, maskW=Colon(), maskH=Colon(), sdsr=1, tdsr=1) =
    ((ml, fitvals, rerrs, denomsum) = fitcomponents(GTW, GTH, W, H; weighted=weighted, clamp=clamp, sdsr=sdsr, tdsr=tdsr);
    (sum(fitvals)/denomsum, ml, fitvals, rerrs))
matchedfitval(X, GTX::AbstractVector, W, H; weighted=true, clamp=false, ordered=false) = (
            (ml, fitvals, rerrs, denomsum) = fitcomponents(X, GTX, W, H; weighted=weighted, clamp=clamp, ordered=ordered);
            (sum(fitvals)/denomsum, ml, fitvals, rerrs)
            )
matchednssd(GTW, GTH, W, H; clamp=false, sdsr=1, tdsr=1) = (
            (ml, mnssds, rerrs) = matchcomponents(GTW, GTH, W, H; clamp=clamp, dsr=dsr, tdsr=tdsr);
            (sum(mnssds)/length(mnssds), ml, mnssds, rerrs)
            )
matchedfitval_clamp(GTW, GTH, W, H; weighted=true, clamp=true, maskW=Colon(), maskH=Colon(), sdsr=1, tdsr=1) =
    ((ml, fitvals, rerrs, denomsum) = fitcomponents_old(GTW, GTH, W, H; weighted=weighted, clamp=clamp, sdsr=sdsr, tdsr=tdsr);
    (sum(fitvals)/denomsum, ml, fitvals, rerrs))

function matchedimg(W, matchlist)
    Wmimg = zeros(size(W,1),length(matchlist))
    for mp in matchlist
        Wmimg[:,mp[1]] = W[:,mp[2]]
    end
    Wmimg
end

function ssdH(ml,gtH,H)
    ssd = 0.
    for (gti, i, invert) in ml
        if i>size(H,2) # no match found
            ssd += sum((gtH[:,gti]).^2)
        else
            ssd += invert ? sum((gtH[:,gti]+H[:,i]).^2) : sum((gtH[:,gti]-H[:,i]).^2)
        end
    end
    ssd
end

# match order with gtW, then W[:,nerorder] is same order with gtW
function matchedorder(ml,ncells)
    neworder = zeros(Int,length(ml))
    for (gti, i) in ml
        neworder[gti]=i
    end
    for i in 1:ncells
        i ∉ neworder && push!(neworder,i)
    end
    neworder
end

function flip2makepos!(W,H; mask=:allpix) # :topNpix
    p = size(W,2)
    for i in 1:p
        (w,h) = view(W,:,i), view(H,i,:)
        pidx = w.>0; nidx = w.<0
        psum = sum(w[pidx]); nsum = -sum(w[nidx])
        if (mask == :topNpix) || ((psum != 0) && (nsum != 0))
            # np = sum(pidx); nn = sum(nidx)
            # pmean = psum/np; nmean = nsum/nn
            # p2mean = sum(w[pidx].^2)/np; n2mean = sum(w[nidx].^2)/nn
            # psvd = sqrt(p2mean - pmean^2); nsvd = sqrt(n2mean - nmean^2)
            # pth = pmean+2psvd; nth = nmean+2nsvd
            # psum = sum(w[w.>pth]); nsum = -sum(w[w.<-nth])
            allmean = (psum-nsum)/length(w); all2mean = sum(w.^2)/length(w)
            allsvd = sqrt(all2mean - allmean^2); pth = allmean+allsvd; nth = allmean-allsvd
            psum = sum(w[w.>pth]); nsum = -sum(w[w.<nth])
        end
        psum < nsum && (w .*= -1; h .*= -1) # just '*=' doesn't work
    end
end

function flip2makepos!(W,H,Mw,Mh)
    p = size(W,2)
    for i in 1:p
        (w,h) = view(W,:,i), view(H,i,:); (mw,mh) = view(Mw,:,i), view(Mh,i,:)
        psum = sum(w[w.>0]); nsum = -sum(w[w.<0])
        psum < nsum && (w .*= -1; mw .*=-1; h .*= -1; mh .*= -1) # just '*=' doesn't work
    end
end

function sortWHslices(W,H)
    pwrs = norm.(eachcol(W)).*norm.(eachrow(H))
    orderindices = sortperm(pwrs, rev=true)
    W[:,orderindices], H[orderindices,:]
end

function getdata(trs::Vector{T},fieldname::Symbol) where T
    iter_num=length(trs)
    val = getfield(trs[1],fieldname)
    vals=typeof(val)[]
    for i in 1:iter_num
        push!(vals,getfield(trs[i],fieldname))
    end
    eltype(vals).(vals)
end

function getdata(trs::Vector{T}) where T
    fldnames = fieldnames(T)
    datadic=Dict{Symbol,Vector}()
    for fldname in fldnames
        push!(datadic,fldname=>getdata(trs,fldname))
    end
    datadic
end

function subtract_baseline(H::AbstractMatrix{T}; q=0.01) where T
    H_nobaseline = similar(H, T)
    for i in Base.axes(H, 1)
        t = quantile(H[i,:], 1-q)
        m = mean(H[i,:])
        b = quantile(H[i,:], q)
        td = abs(t-m); bd = abs(b-m)
        baseline = td < bd ? t : b
        H_nobaseline[i, :] = H[i, :] .- baseline
    end
    H_nobaseline
end

function evaluate_fitvalue(gtW::AbstractArray{T}, gtH::AbstractArray{T}, X, W, H, maskW, maskH;
        delta_f=false, weighted=true, clamp=false, sub_base_q=0.01) where T
    H_nobase = delta_f ? subtract_baseline(H; q=sub_base_q) : H
    if isempty(gtW) || isempty(gtH)
        avgfit, denom = fitd(X[maskW,maskH],W[maskW,:]*H_nobase[:,maskH]); ml = Tuple{Int,Int,Bool}[]
    else
        avgfit, ml = matchedfitval(gtW[maskW,:], gtH[:,maskH], W[maskW,:], H_nobase[:,maskH];
                                weighted=weighted, clamp=clamp)
    end
    avgfit, ml, H_nobase
end

"""
    matched_correlation(gtH, H; maskH=Colon(), allow_sign_flip=false, ml=nothing) -> (avgcorr, ml, corrs)

Ground-truth *spectra* recovery via Hungarian-matched Pearson correlation
("program spectra correlation"). Named `gtH`/`H` (not `gtW`/`W`) to match
the scRNA-seq field's own convention: in cNMF-style analysis (Kotliar et
al.; `code/evaluate.jl`'s `Htrue`/`Hinferred` in the sibling `cNMF`
project), a "program"/GEP is `H` — the `n_components x n_genes` spectra
matrix — while `W` is per-cell *usage*, not a program. NMF has no single
universal W/H axis convention (it depends entirely on how the caller
orients its input `X`: features-as-rows vs. samples-as-rows), so this
name is a deliberate choice, not a claim about any particular solver's
internal layout — **callers are responsible for passing whichever of
their own W/H actually holds the gene-loading/spectra matrix** as `gtH`/
`H` here (see the call sites in LCSVD.jl/NMF.jl's `common.jl` for what
that means concretely for those two solvers).

`gtH` is `n_features x n_true` (one true component per column — this
still follows `AverageFits`'s own features-as-rows/components-as-columns
layout used throughout this module, e.g. [`matchWcomponents`](@ref); only
the parameter *names* follow scRNA-seq's H=spectra convention, not the
matrix orientation), `H` is `n_features x n_inferred` (`n_inferred >=
n_true`). Each `gtH` column is matched to a distinct `H` column by
maximizing correlation (cost = `1-cor`) — the same matching principle
[`matchedfitval`](@ref)/[`fitcomponents`](@ref) use for the (X,W,H)
reconstruction-based `fitval`, but comparing `gtH`/`H` columns directly
(no usage matrix needed) and using Pearson correlation instead of
normalized SSD.

`maskH` restricts both `gtH` and `H` to the same feature subset before
correlating (e.g. a shared HVG mask), matching `evaluate_fitvalue`'s
`maskW`/`maskH` naming for the analogous per-matrix mask.

`ml` supplies a precomputed matched list to *reuse* instead of running a
fresh correlation-maximizing Hungarian: when `ml !== nothing`, no matching
is done here and each correlation is reported on the caller's own
gt↔inferred pairing. Pass the `ml` returned by
[`evaluate_fitvalue`](@ref)/[`matchedfitval`](@ref) (the joint (X,W,H)
reconstruction-fit assignment) to score, say, the usage matrix on exactly
the same component correspondence the spectra/reconstruction fit chose,
rather than a separately-optimized correlation matching. Each `ml` entry is
`(gt_index, inferred_index, invert)` (2-tuples accepted too, `invert`
defaulting to `false`); the incoming `invert` flag is always honored (a pair
matched under inversion scores as `-r = cor(gth_i, -h_j)`), regardless of
`allow_sign_flip` — because the supplied matching already fixed each
component's joint sign, so it is not re-derived here (`allow_sign_flip` only
governs the fresh-Hungarian branch when `ml === nothing`). This is what makes
the two sides of a shared matching sign-consistent: e.g. deciding a
component's sign by its spectra and reporting the usage correlation under that
same sign. Entries whose indices fall outside `R` are skipped.

`allow_sign_flip=true` matches (and should only be used for) factorizations
with no non-negativity constraint on `H`'s companion `W`, where a
component's `(w_i, h_i)` pair carries a free joint sign ambiguity —
flipping the sign of both leaves the reconstruction `w_i*h_i'` (and any
sign-symmetric L1 sparsity penalty) unchanged, so `h_i` and `-h_i` are
equally valid recoveries of true component `i` (e.g. PCB at its
nonnegativity-penalty weight `β=0`; see `cNMF`'s sibling
`code/evaluate.jl`'s `match_programs`/`matched_correlations`, which this
mirrors). With this on, the Hungarian assignment costs on `1-abs(R)`
instead of `1-R` (so a component matched only after sign-flipping isn't
penalized in the assignment itself), and each reported correlation is
`abs(R[i,j])` — the better of `h_j` and `-h_j` against true component `i`.

Returns `avgcorr` (mean matched correlation), `ml` (`Vector{Tuple{Int,
Int,Bool}}`, `(gt_index, inferred_index, invert)` per match — `invert` is
always `false` unless `allow_sign_flip=true` and the match was better
against `-h_j`, kept as a 3-tuple to match `matchWcomponents`/
`matchedimg`/`ssdH`'s shape), and `corrs` (the per-component correlations,
`corrs[i]` matching `ml[i]`).
"""
function matched_correlation(gtH::AbstractMatrix, H::AbstractMatrix; maskH=Colon(), allow_sign_flip::Bool=false, ml=nothing)
    gth = gtH[maskH, :]; h = H[maskH, :]
    R = cor(gth, h)   # n_true x n_inferred: R[i,j] = cor(gth[:,i], h[:,j])
    # A collapsed component (zero-variance / all-zero column, common mid-iteration for
    # noisy or over-complete factorizations) makes cor return NaN; `Hungarian.hungarian`
    # HANGS on a NaN cost matrix. Treat a degenerate pair as zero correlation so the
    # assignment is well-defined and the reported R isn't NaN-poisoned.
    replace!(R, NaN => zero(eltype(R)))
    mlout = Tuple{Int,Int,Bool}[]
    corrs = eltype(R)[]
    if ml === nothing
        # default: pick the matching by a fresh correlation-maximizing Hungarian.
        cost = allow_sign_flip ? 1 .- abs.(R) : 1 .- R
        assignment, _ = hungarian(cost)
        for i in axes(R, 1)
            j = assignment[i]
            j == 0 && continue   # unmatched (only possible when size(H,2) < size(gtH,2))
            r = R[i, j]
            invert = allow_sign_flip && r < 0
            push!(mlout, (i, j, invert))
            push!(corrs, allow_sign_flip ? abs(r) : r)
        end
    else
        # reuse a caller-supplied matched list — e.g. `evaluate_fitvalue`/
        # `matchedfitval`'s `ml`, the joint (X,W,H) reconstruction-fit assignment —
        # instead of running a separate correlation Hungarian, so the correlation is
        # reported on exactly the same gt↔inferred pairing the fit chose. Each `ml`
        # entry is `(gt_index, inferred_index, invert)` (2-tuples also accepted,
        # invert defaulting to false), matching `fitcomponents`/`matchWcomponents`.
        for t in ml
            i, j = t[1], t[2]
            (j == 0 || j > size(R, 2) || i > size(R, 1)) && continue
            invert = length(t) >= 3 ? t[3] : false
            r = R[i, j]
            # honor the incoming pairing's sign flag: the supplied `ml` already fixed
            # the joint sign of each matched (w_j,h_j) pair, so a component matched
            # under inversion aligns as -r (cor(gth_i, -h_j)). `allow_sign_flip` only
            # governs the fresh-Hungarian branch above; here the sign is not re-derived
            # (independently abs-ing each side would let W and H pick inconsistent
            # signs, impossible for one jointly-signed component).
            push!(mlout, (i, j, invert))
            push!(corrs, invert ? -r : r)
        end
    end
    sum(corrs) / length(corrs), mlout, corrs
end

"""
    evaluate_correlation(gtH, H, maskH; allow_sign_flip=false, ml=nothing) -> (avgcorr, ml, corrs)

[`matched_correlation`](@ref) with `evaluate_fitvalue`'s no-ground-truth
fallback: returns `(NaN, Tuple{Int,Int,Bool}[], eltype(H)[])` when `gtH`
is empty, so it can be called unconditionally right alongside
`evaluate_fitvalue` at every trace-recording site regardless of whether
ground truth was supplied. `gtH`/`H` name the spectra/program matrix per
the scRNA-seq convention — see [`matched_correlation`](@ref)'s docstring
for why that isn't necessarily whatever a given solver happens to call
`H` internally; it's on the caller to pass the right matrix. See
[`matched_correlation`](@ref) for `allow_sign_flip`.
"""
function evaluate_correlation(gtH::AbstractArray{T}, H::AbstractArray, maskH; allow_sign_flip::Bool=false, ml=nothing) where T
    isempty(gtH) && return T(NaN), Tuple{Int,Int,Bool}[], T[]
    matched_correlation(gtH, H; maskH=maskH, allow_sign_flip=allow_sign_flip, ml=ml)
end

"""
    wh_correlations(gtW, gtHt, W, Ht; maskW=Colon(), maskH=Colon(), primary=:W,
                    allow_sign_flip=false, weighted=true) -> (wcorr, hcorr)

Matched ground-truth correlation for BOTH factors of an `X ≈ W*Ht'` decomposition,
on a SINGLE shared matching. This is the reusable form of the pattern every
factorization method needs (LCSVD, NMF, CompNMF, …): match components once, then
score both factors on that one correspondence.

`primary` picks what defines the matching; both R values are then reported on that
one pairing (the secondary factor reuses it, and its joint sign decisions, via
[`matched_correlation`](@ref)'s `ml`), so they describe the same ground-truth↔
component correspondence.

- `:W` — the left factor's own correlation-maximizing Hungarian.
- `:H` — the right factor's.
- `:WH` — **both factors jointly**: the Hungarian of [`fitcomponents`](@ref) on the
  rank-1 reconstruction agreement `‖gtWᵢ⊗gtHᵢ − Wⱼ⊗Hⱼ‖²` — the same matching
  [`matchedfitval`](@ref)/`avgfits` already uses. Prefer this when components can be
  near-degenerate in ONE factor: in an over-complete fit (ncomp ≫ ntrue) a surplus
  component can end up a near-copy of a real one in `W` while its `Ht` is quite
  different, which makes the `:W` cost matrix nearly tied. Numerical jitter then flips
  the assignment; `wcorr` barely moves (the pair was tied) but `hcorr` drops sharply,
  as a transient downward spike on an otherwise flat converged trace. Matching on the
  outer product prices the `Ht` disagreement into the cost, so the tie is broken.
  Trade-off: the matching becomes scale-sensitive (`fitd` scores amplitude, unlike
  Pearson R), and `wcorr` is no longer "the best achievable R" but "the R at the
  pairing the reconstruction chose", so it can read slightly lower.

`weighted` applies only to `:WH` (it is [`fitcomponents`](@ref)'s power weighting of
the assignment cost); pass the same value the companion `avgfits`/`wavgfits` call uses
to keep fit and correlation on one pairing. `allow_sign_flip` governs the `:W`/`:H`
Hungarian only — `fitcomponents` does not consider sign flips, so under `:WH` every
match carries `invert=false`.

- `W`   : `features_W × ncomp`  left factor  (columns = components) ; `gtW`  : `features_W × ntrue`
- `Ht`  : `features_H × ncomp`  right factor (`Ht == H'`, columns = components) ; `gtHt` : `features_H × ntrue`
  (`gtHt == permutedims(gtH)` when the right factor's ground truth is stored as `H`)

`maskW`/`maskH` restrict each factor to a feature subset before correlating (see
[`matched_correlation`](@ref)). `allow_sign_flip` is applied to the primary matching
(e.g. for a factorization with no non-negativity constraint, where each `(w_i,h_i)`
pair has a free joint sign); the secondary factor always honors the primary's sign,
keeping the two sides consistent. This module carries no application-specific meaning
for the two factors — which one is `W` vs `Ht` is entirely the caller's orientation.
Returns `(wcorr, hcorr)`.
"""
function wh_correlations(gtW, gtHt, W, Ht; maskW=Colon(), maskH=Colon(),
        primary::Symbol=:W, allow_sign_flip::Bool=false, weighted::Bool=true)
    primary in (:W, :H, :WH) ||
        throw(ArgumentError("primary must be :W, :H or :WH, got $primary"))
    if primary == :WH
        # No ground truth: fall through to the per-factor path, which returns NaN
        # rather than handing `fitcomponents` an empty matrix.
        if isempty(gtW) || isempty(gtHt)
            wcorr, _, _ = evaluate_correlation(gtW,  W,  maskW; allow_sign_flip=allow_sign_flip)
            hcorr, _, _ = evaluate_correlation(gtHt, Ht, maskH; allow_sign_flip=allow_sign_flip)
            return wcorr, hcorr
        end
        # fitcomponents takes the RIGHT factor with components as ROWS
        # (ntrue/ncomp x features_H), so transpose gtHt/Ht; masks are applied by
        # slicing here because `matchedfitval` swallows its own maskW/maskH.
        ml, _, _, _ = fitcomponents(gtW[maskW, :], permutedims(gtHt[maskH, :]),
                                    W[maskW, :],   permutedims(Ht[maskH, :]); weighted=weighted)
        wcorr, _, _ = evaluate_correlation(gtW,  W,  maskW; allow_sign_flip=allow_sign_flip, ml=ml)
        hcorr, _, _ = evaluate_correlation(gtHt, Ht, maskH; allow_sign_flip=allow_sign_flip, ml=ml)
    elseif primary == :H
        hcorr, ml, _ = evaluate_correlation(gtHt, Ht, maskH; allow_sign_flip=allow_sign_flip)
        wcorr, _, _  = evaluate_correlation(gtW,  W,  maskW; allow_sign_flip=allow_sign_flip, ml=ml)
    else
        wcorr, ml, _ = evaluate_correlation(gtW,  W,  maskW; allow_sign_flip=allow_sign_flip)
        hcorr, _, _  = evaluate_correlation(gtHt, Ht, maskH; allow_sign_flip=allow_sign_flip, ml=ml)
    end
    wcorr, hcorr
end

end # module AverageFits
