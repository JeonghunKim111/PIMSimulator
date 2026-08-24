module csc_result_transport_ctrl #(
  parameter int ADDR_W=64, parameter int TAG_W=8,
  parameter int PENDING_WRITE_DEPTH=4,
  parameter int MAX_INFLIGHT_WRITES=2, parameter int MAX_INFLIGHT_READS=2,
  parameter logic [ADDR_W-1:0] RESULT_BASE='0
) (
  input logic clk, input logic rst_n,
  input logic burst_valid, output logic burst_ready,
  input logic [255:0] burst_data, input logic [2:0] burst_record_count,
  input logic producer_done, input logic start_readback,
  output logic write_req_valid, input logic write_req_ready,
  output logic [ADDR_W-1:0] write_req_addr, output logic [255:0] write_req_data,
  output logic [TAG_W-1:0] write_req_tag,
  input logic write_done_valid, input logic [TAG_W-1:0] write_done_tag,
  output logic read_req_valid, input logic read_req_ready,
  output logic [ADDR_W-1:0] read_req_addr, output logic [TAG_W-1:0] read_req_tag,
  input logic read_rsp_valid, input logic [255:0] read_rsp_data,
  input logic [TAG_W-1:0] read_rsp_tag,
  output logic read_data_valid, input logic read_data_ready,
  output logic [255:0] read_data,
  output logic writeback_done, output logic readback_done
);
  localparam int PIW=(PENDING_WRITE_DEPTH<=1)?1:$clog2(PENDING_WRITE_DEPTH);
  logic [255:0] pending_data[PENDING_WRITE_DEPTH];
  logic [2:0] pending_count[PENDING_WRITE_DEPTH];
  logic [PIW-1:0] p_rd,p_wr; logic [$clog2(PENDING_WRITE_DEPTH+1)-1:0] p_count;
  logic [MAX_INFLIGHT_WRITES-1:0] wvalid;
  logic [TAG_W-1:0] wtags[MAX_INFLIGHT_WRITES];
  logic [MAX_INFLIGHT_READS-1:0] rvalid;
  logic [TAG_W-1:0] rtags[MAX_INFLIGHT_READS];
  logic [31:0] issued_writes, completed_writes, issued_reads, completed_reads;
  logic read_phase; logic rsp_hold; logic [255:0] rsp_data_hold;
  integer i; integer wfree,wmatch,rfree,rmatch;
  always_comb begin
    wfree=-1;wmatch=-1;rfree=-1;rmatch=-1;
    for(i=0;i<MAX_INFLIGHT_WRITES;i=i+1) begin
      if(!wvalid[i]&&wfree<0)wfree=i;
      if(wvalid[i]&&wtags[i]==write_done_tag&&wmatch<0)wmatch=i;
    end
    for(i=0;i<MAX_INFLIGHT_READS;i=i+1) begin
      if(!rvalid[i]&&rfree<0)rfree=i;
      if(rvalid[i]&&rtags[i]==read_rsp_tag&&rmatch<0)rmatch=i;
    end
  end
  assign burst_ready=p_count<PENDING_WRITE_DEPTH;
  assign write_req_valid=(p_count!=0)&&(wfree>=0)&&!read_phase;
  assign write_req_data=pending_data[p_rd];
  assign write_req_tag=issued_writes[TAG_W-1:0];
  assign write_req_addr=RESULT_BASE+({32'b0,issued_writes}<<5);
  assign read_req_valid=read_phase&&(issued_reads<completed_writes)&&(rfree>=0);
  assign read_req_tag=issued_reads[TAG_W-1:0];
  assign read_req_addr=RESULT_BASE+({32'b0,issued_reads}<<5);
  assign read_data_valid=rsp_hold; assign read_data=rsp_data_hold;
  always_ff @(posedge clk or negedge rst_n) begin
    if(!rst_n) begin
      p_rd<=0;p_wr<=0;p_count<=0;wvalid<='0;rvalid<='0;
      issued_writes<=0;completed_writes<=0;issued_reads<=0;completed_reads<=0;
      read_phase<=0;rsp_hold<=0;rsp_data_hold<='0;writeback_done<=0;readback_done<=0;
    end else begin
      writeback_done<=0;readback_done<=0;
      if(burst_valid&&burst_ready) begin
        pending_data[p_wr]<=burst_data;pending_count[p_wr]<=burst_record_count;
        p_wr<=(p_wr==PENDING_WRITE_DEPTH-1)?0:p_wr+1'b1;
      end
      if(write_req_valid&&write_req_ready) begin
        wvalid[wfree]<=1;wtags[wfree]<=write_req_tag;issued_writes<=issued_writes+1;
        p_rd<=(p_rd==PENDING_WRITE_DEPTH-1)?0:p_rd+1'b1;
      end
      case({burst_valid&&burst_ready,write_req_valid&&write_req_ready})
        2'b10:p_count<=p_count+1'b1; 2'b01:p_count<=p_count-1'b1; default:begin end
      endcase
      if(write_done_valid&&wmatch>=0) begin wvalid[wmatch]<=0;completed_writes<=completed_writes+1;end
      if(producer_done&&p_count==0&&wvalid=='0&&completed_writes==issued_writes) writeback_done<=1;
      if(start_readback) begin read_phase<=1;issued_reads<=0;completed_reads<=0;end
      if(read_req_valid&&read_req_ready) begin
        rvalid[rfree]<=1;rtags[rfree]<=read_req_tag;issued_reads<=issued_reads+1;
      end
      if(read_rsp_valid&&rmatch>=0&&!rsp_hold) begin
        rvalid[rmatch]<=0;completed_reads<=completed_reads+1;rsp_hold<=1;rsp_data_hold<=read_rsp_data;
      end
      if(rsp_hold&&read_data_ready)rsp_hold<=0;
      if(read_phase&&completed_reads==completed_writes&&issued_reads==completed_writes&&rvalid=='0&&!rsp_hold) begin
        read_phase<=0;readback_done<=1;
      end
    end
  end
endmodule
