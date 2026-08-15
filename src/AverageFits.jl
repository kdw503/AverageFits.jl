module AverageFits

using LinearAlgebra, Statistics, Hungarian, DataStructures, ImageFiltering

export noisefilter, normalizeU!, fitd, matchedfitval, matchedorder, flip2makepos!
export getdata, subtract_baseline, evaluate_fitvalue, matchedWnssda, ssdH

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
        delta_f=false, weighted=true, clamp=false) where T
    H_nobase = delta_f ? subtract_baseline(H; q=0.01) : H
    if isempty(gtW) || isempty(gtH)
        avgfit, denom = fitd(X[maskW,maskH],W[maskW,:]*H_nobase[:,maskH]); ml = Tuple{Int,Int,Bool}[]
    else
        avgfit, ml = matchedfitval(gtW[maskW,:], gtH[:,maskH], W[maskW,:], H_nobase[:,maskH];
                                weighted=weighted, clamp=clamp)
    end
    avgfit, ml, H_nobase
end

end # module AverageFits
