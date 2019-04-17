function Mat_store = Feat_Directionality(fname)

load(fname,'TimeSeries','TS_DataMat','Operations');

for j=1:size(TS_DataMat,2)
   Operations_ID_curdata(j,:) = Operations(j).ID;
end

idx_dir=1;idx_undir = 1;

for i=1:size(TS_DataMat,1)
    if strcmp(TimeSeries(i).Keywords,'dir')
        Directed_set(idx_dir,:) = TS_DataMat(i,:);
        idx_dir=idx_dir+1;
    elseif strcmp(TimeSeries(i).Keywords,'undir')
        Undirected_set(idx_undir,:) = TS_DataMat(i,:);
        idx_undir=idx_undir+1;
    end
end

FDMAT = (sum(Directed_set,1)/size(Directed_set,1)) - (sum(Undirected_set,1)/size(Undirected_set,1));


Mat_store = NaN([1 7873]);

for i=1:length(FDMAT)
   Mat_store(Operations_ID_curdata(i)) = FDMAT(i); 
end

end