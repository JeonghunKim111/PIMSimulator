`timescale 1ns/1ps
module tb_csc_bga;
 localparam N=64;logic clk=0,rst_n=0,iv,ir,drain,dd,ar,ardy,avr;logic[31:0]row,orow;
 logic[15:0]val,aa,ab,av,oval;logic ov,ordy;logic[$clog2(N+1)-1:0]occ;
 logic bv,br,ev,er;logic[3:0]bc;logic[8*32-1:0]brows;logic[8*16-1:0]bvals;
 always #0.5 clk=~clk;
 csc_bga #(.ACC_ENTRIES(N),.COMPARE_W(N),.INPUT_DEPTH(N),.OUTPUT_DEPTH(N)) dut(
  .clk,.rst_n,.in_valid(iv),.in_ready(ir),.in_row(row),.in_value(val),.drain_req(drain),
  .drain_done(dd),.add_req_valid(ar),.add_req_ready(ardy),.add_a(aa),.add_b(ab),
  .add_result_valid(avr),.add_result(av),.out_valid(ov),.out_ready(ordy),.out_row(orow),
  .out_value(oval),.occupancy(occ));
 csc_batch_ingress ingress(.clk,.rst_n,.batch_valid(bv),.batch_ready(br),.batch_count(bc),
  .batch_rows(brows),.batch_values(bvals),.event_valid(ev),.event_ready(er),.event_row(),.event_value());
 task put(input[31:0]r,input[15:0]v);begin @(negedge clk);while(!ir)@(negedge clk);row=r;val=v;iv=1;@(negedge clk);iv=0;end endtask
 integer i,outn;logic saw_merged;logic[15:0]row1_value;
 always_ff @(posedge clk)begin
  if(!rst_n)begin avr<=0;av<=0;outn<=0;saw_merged<=0;row1_value<=0;end else begin
    avr<=0;if(ar&&ardy)begin av<=aa+ab;avr<=1;end
    if(ov&&ordy)begin outn<=outn+1;
      if(orow==1)begin row1_value<=oval;if(oval==7)saw_merged<=1;end end
  end
 end
 initial begin iv=0;row=0;val=0;drain=0;ardy=1;ordy=1;
  bv=0;bc=0;brows=0;bvals=0;er=0;repeat(2)@(negedge clk);rst_n=1;
  bc=1;brows[31:0]=32'h55;bvals[15:0]=16'h1;bv=1;@(negedge clk);bv=0;
  if(!ev)$fatal("batch1 not staged");er=1;@(negedge clk);er=0;
  for(i=0;i<8;i=i+1)begin brows[i*32+:32]=i;bvals[i*16+:16]=i;end
  bc=8;bv=1;@(negedge clk);bv=0;for(i=0;i<8;i=i+1)begin
    if(!ev)$fatal("batch8 missing");er=1;@(negedge clk);
  end er=0;
  for(i=0;i<N;i=i+1)put(i,i+1);repeat(N+5)@(negedge clk);
  if(occ!=N)$fatal("Q64 occupancy %0d",occ);
  put(1,16'd5);repeat(5)@(negedge clk);
  ordy=0;put(100,16'd9);repeat(3)@(negedge clk);if(!ov)$fatal("eviction output absent");
  repeat(2)@(negedge clk);ordy=1;repeat(2)@(negedge clk);
  drain=1;@(negedge clk);drain=0;
  repeat(500)begin @(negedge clk);if(dd)begin if(!saw_merged)$fatal("duplicate row did not merge value=%0d outputs=%0d",row1_value,outn);
    $display("PASS tb_csc_bga outputs=%0d",outn);$finish;end end
  $fatal("drain timeout");end
endmodule
