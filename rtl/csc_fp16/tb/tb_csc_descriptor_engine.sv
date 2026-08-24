`timescale 1ns/1ps
module tb_csc_descriptor_engine;
 logic clk=0,rst_n=0,dv,dr,cv,cr,done,busy;logic[31:0]col,x,nnz,ocol,ox,cid;
 logic[63:0]vo,io,vc,ic;logic[15:0]mask;logic[4:0]lanes;
 always #0.5 clk=~clk;
 csc_descriptor_engine dut(.clk,.rst_n,.desc_valid(dv),.desc_ready(dr),.desc_column_id(col),
  .desc_x_slot(x),.desc_nnz_count(nnz),.desc_value_offset(vo),.desc_index_offset(io),
  .chunk_valid(cv),.chunk_ready(cr),.column_id(ocol),.x_slot(ox),.value_cursor(vc),
  .index_cursor(ic),.lane_mask(mask),.valid_lanes(lanes),.chunk_id(cid),.descriptor_done(done),.busy);
 task load(input[31:0]n);begin @(negedge clk);nnz=n;dv=1;@(negedge clk);dv=0;end endtask
 initial begin dv=0;cr=0;col=7;x=3;vo=64;io=128;nnz=0;
  repeat(2)@(negedge clk);rst_n=1;
  load(0);if(!done)$fatal("zero-NNZ did not complete");
  load(5);if(!cv||lanes!=5||mask!=16'h001f)$fatal("short descriptor mismatch");
  cr=1;@(negedge clk);cr=0;if(!done)$fatal("short completion missing");
  load(16);if(lanes!=16||mask!=16'hffff)$fatal("full chunk mismatch");cr=1;@(negedge clk);cr=0;
  load(19);if(lanes!=16)$fatal("first chunk mismatch");cr=1;@(negedge clk);cr=0;
  if(lanes!=3||mask!=16'h7||vc!=96||ic!=192)$fatal("tail/cursor mismatch");
  cr=1;@(negedge clk);$display("PASS tb_csc_descriptor_engine");$finish;
 end
endmodule
