module csc_transport_top(
 input logic clk,input logic rst_n,input logic in_valid,output logic in_ready,
 input logic[31:0]in_row,input logic[15:0]in_value,input logic flush,input logic producer_done,input logic start_readback,
 output logic write_req_valid,input logic write_req_ready,output logic[63:0]write_req_addr,
 output logic[255:0]write_req_data,output logic[7:0]write_req_tag,
 input logic write_done_valid,input logic[7:0]write_done_tag,
 output logic read_req_valid,input logic read_req_ready,output logic[63:0]read_req_addr,output logic[7:0]read_req_tag,
 input logic read_rsp_valid,input logic[255:0]read_rsp_data,input logic[7:0]read_rsp_tag,
 output logic read_data_valid,input logic read_data_ready,output logic[255:0]read_data,
 output logic writeback_done,output logic readback_done);
 logic rv,rr,bv,br;logic[63:0]record;logic[255:0]burst;logic[2:0]n;
 csc_result_serializer s(.in_valid,.in_ready,.in_row,.in_value,.out_valid(rv),.out_ready(rr),.out_record(record));
 csc_burst_packer p(.clk,.rst_n,.record_valid(rv),.record_ready(rr),.record_data(record),.flush,
   .burst_valid(bv),.burst_ready(br),.burst_data(burst),.burst_record_count(n));
 csc_result_transport_ctrl t(.clk,.rst_n,.burst_valid(bv),.burst_ready(br),.burst_data(burst),
   .burst_record_count(n),.producer_done,.start_readback,.write_req_valid,.write_req_ready,.write_req_addr,
   .write_req_data,.write_req_tag,.write_done_valid,.write_done_tag,.read_req_valid,.read_req_ready,
   .read_req_addr,.read_req_tag,.read_rsp_valid,.read_rsp_data,.read_rsp_tag,.read_data_valid,
   .read_data_ready,.read_data,.writeback_done,.readback_done);
endmodule
