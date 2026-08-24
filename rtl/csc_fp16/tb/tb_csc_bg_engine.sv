`timescale 1ns/1ps
module tb_csc_bg_engine;
 logic clk=0,rst_n=0,dv,dr,rv,rr,rspv,mv,mr,mresv,addq,addr,addv,drain,dd,flush,pdone,sread;
 logic wv,wr,wdv,rdv,rdr,rrv,rdout,rdready,wbd,rbd;logic[31:0]col,x,nnz;logic[63:0]vo,io,ra,wa,rda;
 logic[1:0]rt;logic[7:0]tag,wtag,wdtag,rdtag,rrtag;logic[255:0]mvec,opvec,mres,wdata,rrdata,rdata;
 logic[15:0]msca,mask,opsca,opmask,aa,ab,ares;logic[511:0]rows;
 always #0.5 clk=~clk;
 csc_bg_engine dut(.clk,.rst_n,.desc_valid(dv),.desc_ready(dr),.desc_column_id(col),.desc_x_slot(x),
  .desc_nnz_count(nnz),.desc_value_offset(vo),.desc_index_offset(io),.req_valid(rv),.req_ready(rr),
  .req_type(rt),.req_addr(ra),.req_tag(tag),.rsp_valid(rspv),.rsp_tag(tag),.pim_mul_valid(mv),
  .pim_mul_ready(mr),.pim_mul_vector(mvec),.pim_mul_scalar(msca),.pim_mul_lane_mask(mask),
  .operand_vector(opvec),.operand_scalar(opsca),.row_indices(rows),.operand_lane_mask(opmask),
  .pim_mul_result_valid(mresv),.pim_mul_result(mres),.bga_add_req(addq),.bga_add_ready(addr),
  .bga_add_a(aa),.bga_add_b(ab),.bga_add_result_valid(addv),.bga_add_result(ares),.drain_req(drain),
  .drain_done(dd),.transport_flush(flush),.producer_done(pdone),.start_readback(sread),
  .write_req_valid(wv),.write_req_ready(wr),.write_req_addr(wa),.write_req_data(wdata),.write_req_tag(wtag),
  .write_done_valid(wdv),.write_done_tag(wdtag),.read_req_valid(rdv),.read_req_ready(rdr),
  .read_req_addr(rda),.read_req_tag(rdtag),.read_rsp_valid(rrv),.read_rsp_data(rrdata),
  .read_rsp_tag(rrtag),.read_data_valid(rdout),.read_data_ready(rdready),.read_data(rdata),
  .writeback_done(wbd),.readback_done(rbd));
 initial begin dv=0;col=0;x=0;nnz=0;vo=0;io=0;rr=1;rspv=0;mr=1;mresv=0;opvec=0;opsca=0;
  rows=0;opmask=0;addr=1;addv=0;ares=0;drain=0;flush=0;pdone=0;sread=0;wr=1;wdv=0;wdtag=0;
  rdr=1;rrv=0;rrdata=0;rrtag=0;rdready=1;repeat(3)@(negedge clk);rst_n=1;
  dv=1;@(negedge clk);dv=0;repeat(3)@(negedge clk);
  if(!dr)$fatal("empty descriptor did not return ready");
  $display("PASS tb_csc_bg_engine");$finish;end
endmodule
