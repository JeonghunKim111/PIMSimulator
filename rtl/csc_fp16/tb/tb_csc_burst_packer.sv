`timescale 1ns/1ps
module tb_csc_burst_packer;
 logic clk=0,rst_n=0,rv,rr,flush,bv,br;logic[63:0]rd;logic[255:0]bd;logic[2:0]count;
 always #0.5 clk=~clk;
 csc_burst_packer dut(.clk,.rst_n,.record_valid(rv),.record_ready(rr),.record_data(rd),
  .flush,.burst_valid(bv),.burst_ready(br),.burst_data(bd),.burst_record_count(count));
 task put(input[63:0]v);begin @(negedge clk);while(!rr)@(negedge clk);rd=v;rv=1;@(negedge clk);rv=0;end endtask
 task tail(input integer n);integer i;begin for(i=0;i<n;i=i+1)put(i+1);flush=1;@(negedge clk);flush=0;
   if(!bv||count!=n)$fatal("tail count mismatch");br=1;@(negedge clk);br=0;end endtask
 initial begin rv=0;rd=0;flush=0;br=0;repeat(2)@(negedge clk);rst_n=1;
  tail(1);tail(2);tail(3);put(11);put(12);put(13);put(14);
  if(!bv||count!=4)$fatal("full burst missing");
  repeat(2)@(negedge clk);if(!bv)$fatal("backpressure lost burst");br=1;@(negedge clk);br=0;
  put(21);put(22);put(23);put(24);br=1;@(negedge clk);br=0;
  put(31);flush=1;@(negedge clk);flush=0;br=1;@(negedge clk);
  $display("PASS tb_csc_burst_packer");$finish;end
endmodule
