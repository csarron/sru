
int sru_forward_cuda(THCudaTensor *u_t, THCudaTensor *x_t,THCudaTensor *b_t,
                THCudaTensor *init_t,  THCudaTensor *m_t,  THCudaTensor *h_t, THCudaTensor *c_t,
                int len, int batch, int d, int k, int a_t);

int sru_backward_cuda(THCudaTensor *u_t, THCudaTensor *x_t,
                THCudaTensor *b_t, THCudaTensor *init_t,
                THCudaTensor *m_t, THCudaTensor *c_t,
                THCudaTensor *gh_t, THCudaTensor *gl_t,
                THCudaTensor *gu_t,  THCudaTensor *gx_t,
                THCudaTensor *gb_t,  THCudaTensor *gi_t,
                int len, int batch, int d, int k,int a_t);

int sru_bi_forward_cuda(THCudaTensor *u_t, THCudaTensor *x_t,THCudaTensor *b_t,
                THCudaTensor *init_t,  THCudaTensor *m_t,  THCudaTensor *h_t, THCudaTensor *c_t,
                int len, int batch, int d, int k, int a_t);

int sru_bi_backward_cuda(THCudaTensor *u_t, THCudaTensor *x_t,
                THCudaTensor *b_t, THCudaTensor *init_t,
                THCudaTensor *m_t, THCudaTensor *c_t,
                THCudaTensor *gh_t, THCudaTensor *gl_t,
                THCudaTensor *gu_t,  THCudaTensor *gx_t,
                THCudaTensor *gb_t,  THCudaTensor *gi_t,
                int len, int batch, int d, int k,int a_t);
